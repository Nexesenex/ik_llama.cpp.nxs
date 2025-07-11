#include "ggml.h"
#include "log.h"
#include "common.h"
#include "clip.h"
#include "llava.h"
#include "llama.h"

#include <cstdio>
#include <cstdlib>
#include <vector>

struct llava_context {
    struct clip_ctx * ctx_clip = NULL;
    struct llama_context * ctx_llama = NULL;
    struct llama_model * model = NULL;
};

static void show_additional_info(int /*argc*/, char ** argv) {
    LOG_TEE("\n example usage: %s -m <llava-v1.5-7b/ggml-model-q5_k.gguf> --mmproj <llava-v1.5-7b/mmproj-model-f16.gguf> --image <path/to/an/image.jpg> --image <path/to/another/image.jpg> [--temp 0.1] [-p \"describe the image in detail.\"]\n", argv[0]);
    LOG_TEE("  note: a lower temperature value like 0.1 is recommended for better quality.\n");
}

static void llama_log_callback_logTee(ggml_log_level level, const char * text, void * user_data) {
    (void) level;
    (void) user_data;
    LOG_TEE("%s", text);
}

static struct llama_model * llava_init(gpt_params * params) {
    llama_backend_init();
    llama_numa_init(params->numa);

    llama_model_params model_params = llama_model_params_from_gpt_params(*params);

    llama_model * model = llama_load_model_from_file(params->model.c_str(), model_params);
    if (model == NULL) {
        LOG_TEE("%s: error: unable to load model\n" , __func__);
        return NULL;
    }
    return model;
}

static struct llava_context * llava_init_context(gpt_params * params, llama_model * model) {
    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }

    llama_context_params ctx_params = llama_context_params_from_gpt_params(*params);
    if (params->n_ctx < 2048) {
        // warn user here, "Image processing requires at least 2048 context, setting context to 2048"
        LOG_TEE("%s: warn: Image processing requires at least 2048 context, setting context to 2048\n" , __func__);
        ctx_params.n_ctx = 2048;
    } else {
        ctx_params.n_ctx = params->n_ctx;
    }

    llama_context * ctx_llama = llama_new_context_with_model(model, ctx_params);

    if (ctx_llama == NULL) {
        LOG_TEE("%s: error: failed to create the llama_context\n" , __func__);
        return NULL;
    }

    auto ctx_llava = (struct llava_context *)malloc(sizeof(llava_context));

    ctx_llava->ctx_llama = ctx_llama;
    ctx_llava->model = model;
    return ctx_llava;
}

static void llava_free(struct llava_context * ctx_llava) {
    if (ctx_llava->ctx_clip) {
        clip_free(ctx_llava->ctx_clip);
        ctx_llava->ctx_clip = NULL;
    }

    llama_free(ctx_llava->ctx_llama);
    llama_free_model(ctx_llava->model);
    llama_backend_free();
}

static struct clip_ctx * clip_init_context(gpt_params * params) {
    const char * clip_path = params->mmproj.c_str();

    auto prompt = params->prompt;
    if (prompt.empty()) {
        prompt = "describe the image in detail.";
    }
    auto ctx_clip = clip_model_load(clip_path, /*verbosity=*/ 1);
    return ctx_clip;
}

static bool eval_tokens(struct llama_context * ctx_llama, std::vector<llama_token> tokens, int n_batch, int * n_past) {
    int N = (int) tokens.size();
    for (int i = 0; i < N; i += n_batch) {
        int n_eval = (int) tokens.size() - i;
        if (n_eval > n_batch) {
            n_eval = n_batch;
        }
        if (llama_decode(ctx_llama, llama_batch_get_one(&tokens[i], n_eval, *n_past, 0))) {
            LOG_TEE("%s : failed to eval. token %d/%d (batch size %d, n_past %d)\n", __func__, i, N, n_batch, *n_past);
            return false;
        }
        *n_past += n_eval;
    }
    return true;
}

static bool eval_id(struct llama_context * ctx_llama, int id, int * n_past) {
    std::vector<llama_token> tokens;
    tokens.push_back(id);
    return eval_tokens(ctx_llama, tokens, 1, n_past);
}

static bool eval_string(struct llama_context * ctx_llama, const char* str, int n_batch, int * n_past, bool add_bos){
    std::string              str2     = str;
    std::vector<llama_token> embd_inp = ::llama_tokenize(ctx_llama, str2, add_bos, true);
    return eval_tokens(ctx_llama, embd_inp, n_batch, n_past);
}

static void process_eval_image_embed(struct llava_context * ctx_llava, const struct llava_image_embed * embeds, int n_batch, int * n_past, int idx) {
    float * image_embed = (float *)malloc(clip_embd_nbytes(ctx_llava->ctx_clip));
    std::memcpy(image_embed, embeds->embed + idx * clip_n_patches(ctx_llava->ctx_clip) * clip_n_mmproj_embd(ctx_llava->ctx_clip), clip_embd_nbytes(ctx_llava->ctx_clip));

    auto slice_embed = (llava_image_embed*)malloc(sizeof(llava_image_embed));
    slice_embed->embed = image_embed;
    slice_embed->n_image_pos = clip_n_patches(ctx_llava->ctx_clip);
    llava_eval_image_embed(ctx_llava->ctx_llama, slice_embed, n_batch, n_past);
    llava_image_embed_free(slice_embed);
}

static void process_image(struct llava_context * ctx_llava, struct llava_image_embed * embeds, gpt_params * params, int &n_past) {
    std::string system_prompt;
    int idx = 0;
    int num_image_embeds = embeds->n_image_pos / clip_n_patches(ctx_llava->ctx_clip);
    system_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n";
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
    eval_string(ctx_llava->ctx_llama, (system_prompt+"<image>").c_str(), params->n_batch, &n_past, false);
    process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
    eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
    if (num_image_embeds > 1) {
        size_t num_image_embeds_col = clip_uhd_num_image_embeds_col(ctx_llava->ctx_clip);
        eval_string(ctx_llava->ctx_llama, std::string("<slice>").c_str(), params->n_batch, &n_past, false);
        for (size_t i = 0; i < (num_image_embeds-1)/num_image_embeds_col; ++i) {
            for (size_t j = 0; j < num_image_embeds_col; ++j) {
                eval_string(ctx_llava->ctx_llama, std::string("<image>").c_str(), params->n_batch, &n_past, false);
                process_eval_image_embed(ctx_llava, embeds, params->n_batch, &n_past, idx++);
                eval_string(ctx_llava->ctx_llama, std::string("</image>").c_str(), params->n_batch, &n_past, false);
                if (j == num_image_embeds_col - 1) {
                    eval_string(ctx_llava->ctx_llama, std::string("\n").c_str(), params->n_batch, &n_past, false);
                }
            }
        }
        eval_string(ctx_llava->ctx_llama, std::string("</slice>").c_str(), params->n_batch, &n_past, false);
    }
    LOG_TEE("%s: image token past: %d\n", __func__, n_past);
}

static const char * sample(struct llama_sampling_context * ctx_sampling,
                           struct llama_context * ctx_llama,
                           int * n_past) {
    const llama_token id = llama_sampling_sample(ctx_sampling, ctx_llama, NULL);
    llama_sampling_accept(ctx_sampling, ctx_llama, id, true);
    static std::string ret;
    if (llama_token_is_eog(llama_get_model(ctx_llama), id)) {
        ret = "</s>";
    } else {
        ret = llama_token_to_piece(ctx_llama, id);
    }
    eval_id(ctx_llama, id, n_past);
    return ret.c_str();
}

static struct llava_context * minicpmv_init(gpt_params * params, const std::string & fname, int &n_past){
    auto ctx_clip = clip_init_context(params);
    auto embeds = llava_image_embed_make_with_filename(ctx_clip, params->n_threads, fname.c_str());
    if (!embeds) {
        std::cerr << "error: failed to load image " << fname << ". Terminating\n\n";
        return NULL;
    }

    // process the prompt
    if (params->prompt.empty() && params->interactive == false) {
        LOG_TEE("prompt should be given or interactive mode should be on");
        return NULL;
    }

    auto model = llava_init(params);
    if (model == NULL) {
        fprintf(stderr, "%s: error: failed to init minicpmv model\n", __func__);
        return NULL;
    }
    const int64_t t_llava_init_start_us = ggml_time_us();
    auto ctx_llava = llava_init_context(params, model);
    ctx_llava->ctx_clip = ctx_clip;
    const int64_t t_llava_init_end_us = ggml_time_us();
    float t_llava_init_ms = (t_llava_init_end_us - t_llava_init_start_us) / 1000.0;
    LOG_TEE("\n%s: llava init in %8.2f ms.\n", __func__, t_llava_init_ms);

    const int64_t t_process_image_start_us = ggml_time_us();
    process_image(ctx_llava, embeds, params, n_past);
    const int64_t t_process_image_end_us = ggml_time_us();
    float t_process_image_ms = (t_process_image_end_us - t_process_image_start_us) / 1000.0;
    LOG_TEE("\n%s: llama process image in %8.2f ms.\n", __func__, t_process_image_ms);

    llava_image_embed_free(embeds);
    return ctx_llava;
}

static struct llama_sampling_context * llama_init(struct llava_context * ctx_llava, gpt_params * params, std::string prompt, int &n_past, bool is_first = false){
    std::string user_prompt = prompt;
    if (!is_first) user_prompt = "<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n" + prompt;

    eval_string(ctx_llava->ctx_llama, user_prompt.c_str(), params->n_batch, &n_past, false);
    eval_string(ctx_llava->ctx_llama, "<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", params->n_batch, &n_past, false);
    // generate the response

    LOG_TEE("\n");

    struct llama_sampling_context * ctx_sampling = llama_sampling_init(llama_get_model_vocab(ctx_llava->model),params->sparams);
    return ctx_sampling;
}

static const char * llama_loop(struct llava_context * ctx_llava,struct llama_sampling_context * ctx_sampling, int &n_past){

    const char * tmp = sample(ctx_sampling, ctx_llava->ctx_llama, &n_past);
    return tmp;
}

int main(int argc, char ** argv) {
    ggml_time_init();

    gpt_params params;

    if (!gpt_params_parse(argc, argv, params)) {
        show_additional_info(argc, argv);
        return 1;
    }

#ifndef LOG_DISABLE_LOGS
    log_set_target(log_filename_generator("llava", "log"));
    LOG_TEE("Log start\n");
    log_dump_cmdline(argc, argv);
    llama_log_set(llama_log_callback_logTee, nullptr);
#endif // LOG_DISABLE_LOGS

    if (params.mmproj.empty() || (params.image.empty())) {
        gpt_params_print_usage(argc, argv, params);
        show_additional_info(argc, argv);
        return 1;
    }

    for (auto & image : params.image) {
        int n_past = 0;
        auto ctx_llava = minicpmv_init(&params, image, n_past);

        if (!params.prompt.empty()) {
            LOG_TEE("<user>%s\n", params.prompt.c_str());
            LOG_TEE("<assistant>");
            auto ctx_sampling = llama_init(ctx_llava, &params, params.prompt.c_str(), n_past, true);
            const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
            std::string response = "";
            bool have_tmp = false;
            for (int i = 0; i < max_tgt_len; i++) {
                auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                response += tmp;
                if (strcmp(tmp, "</s>") == 0){
                    if(!have_tmp)continue;
                    else break;
                }
                if (strstr(tmp, "###")) break; // Yi-VL behavior
                have_tmp = true;
                printf("%s", tmp);
                if (strstr(response.c_str(), "<user>")) break; // minicpm-v

                fflush(stdout);
            }
            llama_sampling_free(ctx_sampling);
        }else {
            while (true) {
                LOG_TEE("<user>");
                std::string prompt;
                std::getline(std::cin, prompt);
                LOG_TEE("<assistant>");
                auto ctx_sampling = llama_init(ctx_llava, &params, prompt, n_past, true);
                const int max_tgt_len = params.n_predict < 0 ? 256 : params.n_predict;
                std::string response = "";
                for (int i = 0; i < max_tgt_len; i++) {
                    auto tmp = llama_loop(ctx_llava, ctx_sampling, n_past);
                    response += tmp;
                    if (strcmp(tmp, "</s>") == 0) break;
                    if (strstr(tmp, "###")) break; // Yi-VL behavior
                    printf("%s", tmp);// mistral llava-1.6
                    if (strstr(response.c_str(), "<user>")) break; // minicpm-v
                    fflush(stdout);
                }
                llama_sampling_free(ctx_sampling);
            }
        }
        printf("\n");
        llama_print_timings(ctx_llava->ctx_llama);

        ctx_llava->model = NULL;
        llava_free(ctx_llava);
    }

    return 0;
}
