#!/usr/bin/env python3
"""
llama-perplexity GUI - A graphical interface for llama-perplexity
Profiles saved as JSON, chain loading, drag-and-drop, results logging.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import subprocess
import threading
import json
import os
import re
import time
import ctypes
from pathlib import Path
from ctypes import wintypes

# ---------------------------------------------------------------------------
# Windows drag-and-drop support (no external libs needed)
# ---------------------------------------------------------------------------

try:
    shell32 = ctypes.windll.shell32
    ole32  = ctypes.windll.ole32
    ole32.OleInitialize(None)

    DragQueryFile = shell32.DragQueryFileW
    DragQueryFile.argtypes = [wintypes.HDROP, wintypes.UINT, wintypes.LPWSTR, wintypes.UINT]
    DragQueryFile.restype = wintypes.UINT
    DragFinish = shell32.DragFinishW
    DragFinish.argtypes = [wintypes.HDROP]
    DragAcceptFiles = shell32.DragAcceptFiles
    DragAcceptFiles.argtypes = [wintypes.HWND, wintypes.BOOL]

    _HAVE_DRAG_DROP = True
except Exception:
    _HAVE_DRAG_DROP = False

PROJECT_ROOT = Path(__file__).resolve().parent
PROFILES_DIR = PROJECT_ROOT / "perplexity_profiles"
RESULTS_FILE = PROJECT_ROOT / "results.json"
DEFAULT_BUILD = "x64_Rel_MSVC_CPU"
DEFAULT_EXEC = "llama-perplexity.exe"

# ---------------------------------------------------------------------------
# Parameter definitions: (key, label, type, default, extra_hint)
# type is one of: bool, int, float, str, file, dir, combo
# ---------------------------------------------------------------------------

CACHE_TYPES = "f32|f16|bf16|q8_0|q8_1|q4_0|q4_1|iq4_nl|q5_0|q5_1|q6_0|q8_KV"

PARAM_GROUPS = [
    ("Model & Paths", [
        ("model",          "Model path",            "file",  "",             "(.gguf)"),
        ("prompt_file",    "Prompt data file",       "file",  "",             "(-f, text/CSV)"),
        ("binary_file",    "Binary data file",       "file",  "",             "(-bf, .dat for MC)"),
        ("logdir",         "Log directory",          "dir",   "",             "(--logdir)"),
        ("logits_file",    "Logits file (KL base)",  "file",  "",             "(--save-all-logits)"),
        ("model_alias",    "Model alias",            "str",   "unknown",      "(--alias)"),
        ("prompt",         "Raw prompt text",        "str",   "",             "(-p)"),
        ("in_files",       "Input files (; sep)",   "str",   "",             "(--in-file)"),
        ("lora_adapters",  "LoRA adapters",         "str",   "",             "(--lora)"),
        ("control_vectors","Control vectors",       "str",   "",             "(--control-vector)"),
        ("rpc_servers",    "RPC servers",           "str",   "",             "(--rpc)"),
    ]),
    ("Perplexity Mode", [
        ("ppl_stride",     "PPL stride",            "int",   0,              "(--ppl-stride)"),
        ("ppl_output_type","PPL output type",       "int",   0,              "(0=chunk,1=per-line)"),
        ("n_chunks",       "Max chunks",            "int",   -1,             "(-1 = all)"),
        ("hellaswag",      "HellaSwag score",       "bool",  False,          ""),
        ("hellaswag_tasks","HellaSwag tasks",       "int",   400,            ""),
        ("winogrande",     "Winogrande score",      "bool",  False,          ""),
        ("winogrande_tasks","Winogrande tasks",     "int",   0,              "(0=all)"),
        ("multiple_choice","Multiple choice score", "bool",  False,          ""),
        ("multiple_choice_tasks","MC tasks",        "int",   0,              "(0=all)"),
        ("kl_divergence",  "KL divergence",         "bool",  False,          ""),
        ("tinylog",        "Tiny log mode",         "bool",  False,          "(--tinylog)"),
    ]),
    ("Compute", [
        ("n_threads",      "Threads (generation)",  "int",   0,              "(-t, 0=auto)"),
        ("n_threads_batch","Threads (batch)",       "int",   -1,             "(-tb, -1=use -t)"),
        ("n_batch",        "Logical batch size",    "int",   2048,           "(-b)"),
        ("n_ubatch",       "Physical batch size",   "int",   512,            "(-ub)"),
        ("n_ctx",          "Context size",          "int",   512,            "(-c)"),
        ("n_keep",         "Tokens to keep",        "int",   0,              "(--keep)"),
        ("n_parallel",     "Parallel sequences",    "int",   1,              "(--parallel)"),
        ("n_sequences",    "Sequences to decode",   "int",   1,              "(-ns)"),
        ("cont_batching",  "Continuous batching",   "bool",  True,           "(-cb)"),
        ("ctx_shift",      "Context shift",         "bool",  True,           "(--context-shift)"),
    ]),
    ("GPU", [
        ("n_gpu_layers",   "GPU layers",            "int",   -1,             "(-ngl, -1=default)"),
        ("main_gpu",       "Main GPU",              "int",   0,              "(-mg)"),
        ("max_gpu_per_split","Max GPU per split",   "int",   0,              "(0=auto)"),
        ("tensor_split",   "Tensor split (; sep)",  "str",   "",             "(-ts, comma-separated)"),
        ("split_mode",     "Split mode",            "combo", "layer",        "(layer|graph|tenpar|none)"),
        ("devices",        "GPU devices",            "str",   "",             "(--device CUDA0,...)"),
        ("no_kv_offload",  "No KV offload",         "bool",  False,          ""),
        ("cuda_params",    "CUDA params",           "str",   "",             "(-cuda key=val,...)"),
        ("split_output_tensor","Split output tensor","bool", False,          ""),
        ("split_mode_tensor_parallel_scheduling","Force TP scheduling","bool", False, ""),
        ("scheduler_async","Async scheduler",       "bool",  False,          ""),
        ("graph_attn_precision","Graph attn precision","str","f16",          "(-gap)"),
        ("pipeline",       "Pipeline mode",         "int",   0,              "(0=off,1=lookahead,2=selfcopy)"),
        ("sched_max_copies","Sched max copies",     "int",   -1,             "(-smc, -1=default)"),
        ("split_adjust_step_frequency","Split adjust freq","float",0.5,      "(-sasf)"),
        ("worst_graph_tokens","Worst graph tokens", "int",   0,              "(-wgt)"),
        ("ncmoe",          "CPU MoE layers",        "int",   0,              "(-cmoe)"),
        ("fit",            "Auto-fit model",         "bool",  False,         "(--fit)"),
        ("fit_margin",     "Fit margin (MiB)",       "int",   0,             "(--fit-margin)"),
        ("indexer_cache_type_k","Indexer K-cache type","combo","f16",        f"({CACHE_TYPES})"),
    ]),
    ("Optimization", [
        ("flash_attn",     "Flash attention",       "bool",  True,           "(-fa)"),
        ("mla_attn",       "MLA mode",              "int",   3,              "(0=std,1=K+VT,2=K,3=best)"),
        ("attn_max_batch", "Attention max batch",   "int",   0,              ""),
        ("fused_moe_up_gate","Fused MoE up*gate",   "bool",  True,           ""),
        ("fused_up_gate",  "Fused up*gate",         "bool",  True,           ""),
        ("fused_mmad",     "Fused mul+multiadd",    "bool",  True,           ""),
        ("graph_reuse",    "Graph reuse",           "bool",  True,           "(-gr)"),
        ("grouped_expert_routing","Grouped expert routing","bool",False,     "(-ger)"),
        ("merge_qkv",      "Merge Q,K,V",           "bool",  False,          "(-mqkv)"),
        ("merge_up_gate_exps","Merge up/gate exps", "bool",  False,          "(-muge)"),
        ("k_cache_hadamard","K-cache Hadamard",     "bool",  False,          "(-khad)"),
        ("v_cache_hadamard","V-cache Hadamard",     "bool",  False,          "(-vhad)"),
        ("rope_cache",     "RoPE cache",            "bool",  False,          "(-rcache)"),
        ("cache_type_k",   "K-cache type",          "combo", "f16",          f"({CACHE_TYPES})"),
        ("cache_type_v",   "V-cache type",          "combo", "f16",          f"({CACHE_TYPES})"),
        ("reduce_type",    "Graph reduce type",     "combo", "f16",          "(f32|f16|bf16|q8_0|q8_1|hybrid)"),
        ("dsa",            "DSA sparse attention",  "bool",  False,          "(--dsa)"),
        ("dsa_top_k",      "DSA top-k override",    "int",   -1,             ""),
        ("has_mtp",        "Enable MTP",            "bool",  False,          ""),
        ("fused_delta_net","Fused delta net tokens","int",   0,              "(0=off)"),
        ("defrag_thold",   "Defrag threshold",      "float", -1.0,           "(-dt)"),
        ("grp_attn_n",     "Group attention N",     "int",   1,              ""),
        ("grp_attn_w",     "Group attention W",     "int",   512,            ""),
        ("rope_freq_base", "RoPE freq base",        "float", 0.0,            ""),
        ("rope_freq_scale","RoPE freq scale",       "float", 0.0,            ""),
        ("yarn_ext_factor","YaRN ext factor",       "float", -1.0,           ""),
        ("yarn_attn_factor","YaRN attn factor",     "float", -1.0,           ""),
        ("yarn_beta_fast", "YaRN beta fast",        "float", -1.0,           ""),
        ("yarn_beta_slow", "YaRN beta slow",        "float", -1.0,           ""),
        ("yarn_orig_ctx",  "YaRN orig ctx",         "int",   0,              ""),
        ("min_experts",    "Min experts (val,thold)","int",   -1,             "(-ser val,thold)"),
        ("defer_experts",  "Defer experts",         "bool",  False,          ""),
        ("dump_kv_cache",  "Dump KV cache",         "bool",  False,          "(-dkvc)"),
    ]),
    ("Advanced", [
        ("seed",           "RNG seed",              "int",   0,              "(-s, 0=time-based)"),
        ("numa",           "NUMA strategy",         "combo", "disabled",     "(disabled|distribute|isolate|numactl)"),
        ("n_print",        "Print token count",     "int",   -1,             "(-ptc)"),
        ("dry_run",        "Dry run (skip tensors)","bool",  False,          "(-dr)"),
        ("ignore_unknown", "Ignore unknown tokens", "bool",  True,           "(-iu)"),
        ("ignore_eos",     "Ignore EOS",            "bool",  False,          ""),
        ("escape",         "Escape sequences",      "bool",  True,           ""),
        ("use_color",      "Use color",             "bool",  False,          "(-co)"),
        ("verbose_prompt", "Verbose prompt",        "bool",  False,          ""),
        ("warmup",         "Warmup",                "bool",  True,           ""),
        ("batch_warmup",   "Batch warmup",          "bool",  False,          ""),
        ("use_mmap",       "Use mmap",              "bool",  True,           ""),
        ("use_mlock",      "Use mlock",             "bool",  False,          ""),
        ("check_tensors",  "Check tensors",         "bool",  False,          ""),
        ("repack_tensors", "Repack tensors",        "bool",  False,          "(-rtr)"),
        ("validate_quants","Validate quants",       "bool",  False,          "(-vq)"),
        ("only_active_exps","Only active experts",  "bool",  True,           ""),
    ]),
]


def default_params():
    d = {}
    for _group, params in PARAM_GROUPS:
        for key, _label, ptype, default, _hint in params:
            d[key] = default
    d["_build_dir"] = DEFAULT_BUILD
    d["_exec_name"] = DEFAULT_EXEC
    d["_profile_name"] = ""
    return d


class CollapsibleFrame(ttk.LabelFrame):
    def __init__(self, parent, text, **kwargs):
        super().__init__(parent, text=text, **kwargs)
        self.toggle_btn = ttk.Button(self, text="[-]", width=3, command=self.toggle)
        self.toggle_btn.place(relx=1.0, x=-5, y=2, anchor="ne")
        self.content_frame = ttk.Frame(self)
        self.content_frame.pack(fill="x", expand=True, padx=5, pady=5)
        self._collapsed = False

    def toggle(self):
        if self._collapsed:
            self.content_frame.pack(fill="x", expand=True, padx=5, pady=5)
            self.toggle_btn.config(text="[-]")
            self._collapsed = False
        else:
            self.content_frame.pack_forget()
            self.toggle_btn.config(text="[+]")
            self._collapsed = True


class PerplexityGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("llama-perplexity GUI")
        self.root.geometry("1600x900")

        self.params = default_params()
        self.widgets = {}
        self.chain_list = []
        self.process = None

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._build_ui()
        self._load_results()

    def _on_close(self):
        """Kill the running process and destroy the window."""
        if self.process and self.process.poll() is None:
            self.process.terminate()
        self.root.destroy()

    # ------------------------------------------------------------------
    # UI Construction
    # ------------------------------------------------------------------

    def _build_ui(self):
        top_frame = ttk.Frame(self.root)
        top_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(top_frame, text="Perplexity GUI", font=("Segoe UI", 14, "bold")).pack(side="left")

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(top_frame, textvariable=self.status_var, foreground="gray").pack(side="right", padx=8)

        # --- Build config ---
        build_frame = ttk.LabelFrame(self.root, text="Build Configuration", padding=5)
        build_frame.pack(fill="x", padx=8, pady=2)

        ttk.Label(build_frame, text="Build:").grid(row=0, column=0, sticky="w", padx=2)
        self._build_dir_var = tk.StringVar()
        self._build_combo = ttk.Combobox(build_frame, textvariable=self._build_dir_var, width=50, state="readonly")
        self._build_combo.grid(row=0, column=1, sticky="ew", padx=2)
        self._refresh_builds()

        ttk.Label(build_frame, text="Executable:").grid(row=1, column=0, sticky="w", padx=2)
        self._exec_var = tk.StringVar(value=DEFAULT_EXEC)
        ttk.Entry(build_frame, textvariable=self._exec_var, width=50).grid(row=1, column=1, sticky="ew", padx=2)

        build_frame.columnconfigure(1, weight=1)

        # --- Profile bar ---
        profile_frame = ttk.Frame(self.root)
        profile_frame.pack(fill="x", padx=8, pady=4)

        ttk.Label(profile_frame, text="Profile:").pack(side="left")
        self._profile_var = tk.StringVar()
        self._profile_combo = ttk.Combobox(profile_frame, textvariable=self._profile_var, width=30, state="readonly")
        self._profile_combo.pack(side="left", padx=4)
        self._profile_combo.bind("<<ComboboxSelected>>", self._on_profile_select)
        ttk.Button(profile_frame, text="Save", command=self._save_profile).pack(side="left", padx=2)
        ttk.Button(profile_frame, text="Save As...", command=self._save_profile_as).pack(side="left", padx=2)
        ttk.Button(profile_frame, text="Delete", command=self._delete_profile).pack(side="left", padx=2)
        ttk.Button(profile_frame, text="Reset", command=self._reset_params).pack(side="left", padx=2)

        # --- Benchmarks quick-select ---
        bench_frame = ttk.LabelFrame(self.root, text="Benchmarks", padding=5)
        bench_frame.pack(fill="x", padx=8, pady=2)
        self._bench_var = tk.StringVar()
        self._bench_combo = ttk.Combobox(bench_frame, textvariable=self._bench_var, width=60, state="readonly")
        self._bench_combo.pack(side="left", padx=2)
        ttk.Button(bench_frame, text="Apply", command=self._apply_benchmark).pack(side="left", padx=2)
        ttk.Button(bench_frame, text="Refresh", command=self._refresh_benchs).pack(side="left", padx=2)
        self._refresh_benchs()

        # --- Two-column scrollable parameter area ---
        outer_frame = ttk.Frame(self.root)
        outer_frame.pack(fill="both", expand=True, padx=8, pady=2)

        canvas = tk.Canvas(outer_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer_frame, orient="vertical", command=canvas.yview)
        scrollable = ttk.Frame(canvas)

        scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        def _on_mousewheel(event):
            canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_mousewheel)

        # Three columns inside the scrollable area
        cols_frame = ttk.Frame(scrollable)
        cols_frame.pack(fill="x", expand=True)

        col1 = ttk.Frame(cols_frame)
        col2 = ttk.Frame(cols_frame)
        col3 = ttk.Frame(cols_frame)
        col1.pack(side="left", fill="y", anchor="n", expand=True, padx=(0, 3))
        col2.pack(side="left", fill="y", anchor="n", expand=True, padx=(3, 3))
        col3.pack(side="left", fill="y", anchor="n", expand=True, padx=(3, 0))

        # 6 groups: left=model+GPU, center=perplexity+compute+advanced, right=optimization
        self._build_param_fields(col1, [PARAM_GROUPS[0], PARAM_GROUPS[3]])
        self._build_param_fields(col2, [PARAM_GROUPS[1], PARAM_GROUPS[2], PARAM_GROUPS[5]])
        self._build_param_fields(col3, [PARAM_GROUPS[4]])

        # --- Chain loader ---
        chain_frame = ttk.LabelFrame(self.root, text="Chain Loader (drop .json profile files here)", padding=5)
        chain_frame.pack(fill="x", padx=8, pady=4)

        chain_top = ttk.Frame(chain_frame)
        chain_top.pack(fill="x")
        ttk.Button(chain_top, text="Add Profile", command=self._chain_add).pack(side="left", padx=2)
        ttk.Button(chain_top, text="Remove Selected", command=self._chain_remove).pack(side="left", padx=2)
        ttk.Button(chain_top, text="Clear Chain", command=self._chain_clear).pack(side="left", padx=2)
        ttk.Button(chain_top, text="Move Up", command=self._chain_up).pack(side="left", padx=2)
        ttk.Button(chain_top, text="Move Down", command=self._chain_down).pack(side="left", padx=2)

        self._chain_listbox = tk.Listbox(chain_frame, height=4, selectmode="single")
        self._chain_listbox.pack(fill="x", pady=4)
        self._chain_listbox.bind("<Delete>", lambda e: self._chain_remove())

        # drag-drop support
        if _HAVE_DRAG_DROP:
            hwnd = self._chain_listbox.winfo_id()
            DragAcceptFiles(hwnd, True)
            self._chain_listbox_drop_msg = None
            self._bind_drag_drop_message()
        self._chain_listbox.bind("<Button-3>", self._chain_paste_file)
        self._chain_listbox.bind("<Button-2>", self._chain_paste_file)

        # --- Launch / Output ---
        action_frame = ttk.Frame(self.root)
        action_frame.pack(fill="x", padx=8, pady=4)

        self._launch_btn = ttk.Button(action_frame, text="LAUNCH", command=self._launch)
        self._launch_btn.pack(side="left", padx=4)
        ttk.Button(action_frame, text="Stop", command=self._stop_process).pack(side="left", padx=4)

        self._chain_launch_btn = ttk.Button(action_frame, text="Launch Chain", command=self._launch_chain)
        self._chain_launch_btn.pack(side="left", padx=4)

        self._result_var = tk.StringVar(value="")
        ttk.Label(action_frame, textvariable=self._result_var, foreground="blue").pack(side="right", padx=8)

        # --- Output log ---
        out_frame = ttk.LabelFrame(self.root, text="Output", padding=5)
        out_frame.pack(fill="both", expand=True, padx=8, pady=4)

        out_top = ttk.Frame(out_frame)
        out_top.pack(fill="x")
        ttk.Button(out_top, text="Clear", command=self._clear_output).pack(side="left", padx=2)
        ttk.Button(out_top, text="View Results", command=self._view_results).pack(side="left", padx=2)

        self._output_text = tk.Text(out_frame, height=8, wrap="word", state="normal",
            bg="#0C0C0C", fg="#E0E0E0", insertbackground="#E0E0E0",
            relief="flat", borderwidth=0, font=("Consolas", 10))
        out_scroll = ttk.Scrollbar(out_frame, orient="vertical", command=self._output_text.yview)
        self._output_text.configure(yscrollcommand=out_scroll.set)
        self._output_text.tag_configure("stdout", foreground="#CCCCCC")
        self._output_text.tag_configure("stderr", foreground="#FF6B6B")
        self._output_text.tag_configure("cmd", foreground="#569CD6")
        self._output_text.tag_configure("info", foreground="#6A9955")
        self._output_text.tag_configure("ppl", foreground="#DCDCAA")
        self._output_text.tag_configure("error", foreground="#F44747", font=("Consolas", 10, "bold"))
        self._output_text.tag_configure("result", foreground="#CE9178", font=("Consolas", 10, "bold"))
        self._output_text.pack(side="left", fill="both", expand=True)
        out_scroll.pack(side="right", fill="y")

        self._refresh_profiles()

    def _build_param_fields(self, parent, groups):
        for group_name, params in groups:
            cf = CollapsibleFrame(parent, text=group_name)
            cf.pack(fill="x", padx=4, pady=3)

            for idx, (key, label, ptype, default, hint) in enumerate(params):
                row_frame = ttk.Frame(cf.content_frame)
                row_frame.pack(fill="x", pady=1)

                lbl_text = label + (" " + hint if hint else "")
                lbl = ttk.Label(row_frame, text=lbl_text, width=40, anchor="w")
                lbl.pack(side="left")

                if ptype == "bool":
                    var = tk.BooleanVar(value=default)
                    cb = ttk.Checkbutton(row_frame, variable=var)
                    cb.pack(side="left", padx=4)
                    self.widgets[key] = ("bool", var, cb)
                elif ptype == "combo":
                    var = tk.StringVar(value=str(default))
                    values = [x.strip() for x in hint.split("(")[-1].rstrip(")").split("|")] if "|" in hint else []
                    if not values:
                        values = [str(default), "layer", "graph", "tenpar", "none", "disabled", "distribute", "isolate", "numactl"]
                    cb = ttk.Combobox(row_frame, textvariable=var, values=values, width=20, state="readonly")
                    cb.pack(side="left", padx=4)
                    self.widgets[key] = ("combo", var, cb)
                elif ptype in ("file", "dir"):
                    var = tk.StringVar(value=str(default))
                    e = ttk.Entry(row_frame, textvariable=var, width=50)
                    e.pack(side="left", padx=2, fill="x", expand=True)
                    browse_text = "File" if ptype == "file" else "Dir"
                    btn = ttk.Button(row_frame, text=browse_text,
                                     command=lambda k=key, t=ptype: self._browse_param(k, t))
                    btn.pack(side="left", padx=2)
                    self.widgets[key] = ("str", var, e)
                elif ptype == "int":
                    var = tk.StringVar(value=str(default))
                    e = ttk.Entry(row_frame, textvariable=var, width=16)
                    e.pack(side="left", padx=4)
                    self.widgets[key] = ("int", var, e)
                elif ptype == "float":
                    var = tk.StringVar(value=str(default))
                    e = ttk.Entry(row_frame, textvariable=var, width=16)
                    e.pack(side="left", padx=4)
                    self.widgets[key] = ("float", var, e)
                else:
                    var = tk.StringVar(value=str(default))
                    e = ttk.Entry(row_frame, textvariable=var, width=50)
                    e.pack(side="left", padx=4, fill="x", expand=True)
                    self.widgets[key] = ("str", var, e)

    # ------------------------------------------------------------------
    # Parameter helpers
    # ------------------------------------------------------------------

    def _get_param_value(self, key):
        entry = self.widgets.get(key)
        if entry is None:
            return ""
        ptype, var, _widget = entry
        if ptype == "bool":
            return var.get()
        raw = var.get().strip()
        if ptype == "int":
            try:
                return int(raw)
            except ValueError:
                return 0
        elif ptype == "float":
            try:
                return float(raw)
            except ValueError:
                return 0.0
        else:
            return raw

    def _set_param_value(self, key, value):
        entry = self.widgets.get(key)
        if entry is None:
            return
        ptype, var, widget = entry
        if ptype == "bool":
            var.set(bool(value))
        elif ptype == "int":
            var.set(str(int(value)))
        elif ptype == "float":
            var.set(str(float(value)))
        else:
            var.set(str(value))

    def _read_all_params(self):
        d = {}
        for group_name, params in PARAM_GROUPS:
            for key, _label, _ptype, _default, _hint in params:
                d[key] = self._get_param_value(key)
        d["_build_dir"] = self._build_dir_var.get()
        d["_exec_name"] = self._exec_var.get()
        d["_profile_name"] = self._profile_var.get()
        return d

    def _apply_params(self, d):
        for group_name, params in PARAM_GROUPS:
            for key, _label, _ptype, _default, _hint in params:
                if key in d:
                    self._set_param_value(key, d[key])
        if "_build_dir" in d:
            val = str(d["_build_dir"])
            # strip full path to just the dir name for old-style profiles
            p = Path(val)
            if p.parent.name == "bin":
                val = p.parent.parent.name
            elif p.name != val:
                val = p.name
            # verify it's in the combo; if not, pick first available
            names = self._build_combo["values"]
            if val not in names and names:
                val = names[0]
            self._build_dir_var.set(val)
        if "_exec_name" in d:
            self._exec_var.set(str(d["_exec_name"]))
        if "_profile_name" in d:
            self._profile_var.set(str(d["_profile_name"]))

    def _refresh_builds(self):
        """Populate the build combo from out/build/ subdirectories that have a bin/ folder."""
        build_root = PROJECT_ROOT / "out" / "build"
        names = []
        if build_root.is_dir():
            for c in sorted(build_root.iterdir()):
                if c.is_dir():
                    bin_dir = c / "bin"
                    if bin_dir.is_dir():
                        names.append(c.name)
        self._build_combo["values"] = names
        self._build_combo.configure(height=len(names))
        if DEFAULT_BUILD in names:
            self._build_dir_var.set(DEFAULT_BUILD)
        elif names:
            self._build_dir_var.set(names[0])

    def _refresh_benchs(self):
        """Populate the benchmarks combo from benchs/ directory."""
        bench_root = PROJECT_ROOT / "benchs"
        files = []
        if bench_root.is_dir():
            for f in sorted(bench_root.iterdir()):
                if f.is_file():
                    files.append(f.name)
        self._bench_combo["values"] = files

    def _apply_benchmark(self):
        name = self._bench_var.get()
        if not name:
            return
        path = str(PROJECT_ROOT / "benchs" / name)
        ext = Path(name).suffix.lower()
        # Reset benchmark mode flags
        self._set_param_value("multiple_choice", False)
        self._set_param_value("winogrande", False)
        self._set_param_value("hellaswag", False)
        if ext == ".dat":
            # Binary multiple-choice file
            self._set_param_value("binary_file", path)
            self._set_param_value("prompt_file", "")
            self._set_param_value("multiple_choice", True)
        elif ext == ".csv":
            # Winogrande CSV
            self._set_param_value("prompt_file", path)
            self._set_param_value("binary_file", "")
            self._set_param_value("winogrande", True)
        else:
            # Raw text file
            self._set_param_value("prompt_file", path)
            self._set_param_value("binary_file", "")
        self._log(f"Benchmark loaded: {name}\n", "info")

    # ------------------------------------------------------------------
    # Profile system
    # ------------------------------------------------------------------

    def _profile_path(self, name):
        return PROFILES_DIR / f"{name}.json"

    def _refresh_profiles(self):
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        files = sorted(PROFILES_DIR.glob("*.json"))
        names = [f.stem for f in files]
        self._profile_combo["values"] = names
        if self._profile_var.get() not in names:
            self._profile_var.set("")

    def _save_profile(self):
        name = self._profile_var.get().strip()
        if not name:
            self._save_profile_as()
            return
        self._do_save(name)

    def _save_profile_as(self):
        name = tk.simpledialog.askstring("Save Profile", "Profile name:", parent=self.root)
        if not name:
            return
        name = name.strip().replace(".json", "")
        if not name:
            return
        self._profile_var.set(name)
        self._do_save(name)

    def _do_save(self, name):
        data = self._read_all_params()
        path = self._profile_path(name)
        PROFILES_DIR.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        self._refresh_profiles()
        self._log(f"Profile saved: {name}\n")

    def _on_profile_select(self, event=None):
        name = self._profile_var.get()
        if not name:
            return
        path = self._profile_path(name)
        if not path.exists():
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self._apply_params(data)
        self._log(f"Profile loaded: {name}\n")

    def _delete_profile(self):
        name = self._profile_var.get()
        if not name:
            return
        if not messagebox.askyesno("Delete", f"Delete profile '{name}'?", parent=self.root):
            return
        path = self._profile_path(name)
        if path.exists():
            path.unlink()
        self._refresh_profiles()
        self._profile_var.set("")
        self._log(f"Profile deleted: {name}\n")

    def _reset_params(self):
        if not messagebox.askyesno("Reset", "Reset all parameters to defaults?", parent=self.root):
            return
        self._apply_params(default_params())
        self._log("Parameters reset to defaults.\n")

    # ------------------------------------------------------------------
    # Browse helpers
    # ------------------------------------------------------------------

    def _browse_param(self, key, ptype):
        if ptype == "file":
            path = filedialog.askopenfilename(title=f"Select file for {key}",
                                               initialdir=str(PROJECT_ROOT))
        else:
            path = filedialog.askdirectory(title=f"Select directory for {key}",
                                           initialdir=str(PROJECT_ROOT))
        if path:
            entry = self.widgets.get(key)
            if entry:
                _ptype, var, _widget = entry
                var.set(path)

    # ------------------------------------------------------------------
    # Chain loader
    # ------------------------------------------------------------------

    def _chain_add(self):
        files = filedialog.askopenfilenames(title="Select profile files",
                                            initialdir=str(PROFILES_DIR),
                                            filetypes=[("JSON profiles", "*.json")])
        for f in files:
            name = os.path.basename(f)
            if f not in self.chain_list:
                self.chain_list.append(f)
                self._chain_listbox.insert("end", name)

    def _chain_remove(self):
        sel = self._chain_listbox.curselection()
        if sel:
            idx = sel[0]
            self.chain_list.pop(idx)
            self._chain_listbox.delete(idx)

    def _chain_clear(self):
        self.chain_list.clear()
        self._chain_listbox.delete(0, "end")

    def _chain_up(self):
        sel = self._chain_listbox.curselection()
        if sel and sel[0] > 0:
            idx = sel[0]
            self.chain_list[idx], self.chain_list[idx-1] = self.chain_list[idx-1], self.chain_list[idx]
            self._chain_listbox.delete(idx)
            self._chain_listbox.insert(idx-1, os.path.basename(self.chain_list[idx-1]))
            self._chain_listbox.selection_set(idx-1)

    def _chain_down(self):
        sel = self._chain_listbox.curselection()
        if sel and sel[0] < len(self.chain_list) - 1:
            idx = sel[0]
            self.chain_list[idx], self.chain_list[idx+1] = self.chain_list[idx+1], self.chain_list[idx]
            self._chain_listbox.delete(idx)
            self._chain_listbox.insert(idx+1, os.path.basename(self.chain_list[idx+1]))
            self._chain_listbox.selection_set(idx+1)

    def _chain_paste_file(self, event):
        try:
            self.root.clipboard_get()
        except tk.TclError:
            return
        path = self.root.clipboard_get().strip()
        if path.endswith(".json") and os.path.isfile(path):
            abs_path = os.path.abspath(path)
            if abs_path not in self.chain_list:
                self.chain_list.append(abs_path)
                self._chain_listbox.insert("end", os.path.basename(abs_path))

    def _bind_drag_drop_message(self):
        """Register a Windows message handler for WM_DROPFILES (0x233)."""
        WM_DROPFILES = 0x0233
        GWL_WNDPROC = -4

        def wnd_proc(hwnd, msg, wparam, lparam):
            if msg == WM_DROPFILES:
                hdrop = wintypes.HDROP(wparam)
                count = DragQueryFile(hdrop, -1, None, 0)
                for i in range(count):
                    buf = ctypes.create_unicode_buffer(260)
                    DragQueryFile(hdrop, i, buf, 260)
                    path = buf.value
                    if path.endswith(".json") and os.path.isfile(path):
                        self.root.after(0, self._chain_add_path, path)
                DragFinish(hdrop)
                return 0
            return ctypes.windll.user32.DefWindowProcW(hwnd, msg, wparam, lparam)

        hwnd = self._chain_listbox.winfo_id()
        wndproc_type = ctypes.WINFUNCTYPE(ctypes.c_int64, wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
        self._chain_listbox_wndproc = wndproc_type(wnd_proc)
        user32 = ctypes.windll.user32
        try:
            set_ptr = user32.SetWindowLongPtrW
        except AttributeError:
            set_ptr = user32.SetWindowLongW
        set_ptr(hwnd, GWL_WNDPROC, ctypes.cast(self._chain_listbox_wndproc, ctypes.c_int64))

    def _chain_add_path(self, path):
        abs_path = os.path.abspath(path)
        if abs_path not in self.chain_list:
            self.chain_list.append(abs_path)
            self._chain_listbox.insert("end", os.path.basename(abs_path))

    # ------------------------------------------------------------------
    # Launch logic
    # ------------------------------------------------------------------

    def _build_cmd(self):
        build_name = self._build_dir_var.get().strip()
        if not build_name:
            self._log("ERROR: no build selected\n")
            return None
        build_dir = PROJECT_ROOT / "out" / "build" / build_name / "bin"
        exec_name = self._exec_var.get().strip()
        exe_path = build_dir / exec_name

        if not exe_path.exists():
            exe_path = build_dir / "llama-perplexity.exe"
        if not exe_path.exists():
            self._log(f"ERROR: executable not found in {build_dir}\n")
            return None

        cmd = [str(exe_path)]

        # Always force --logdir (default perplexity_logs subdir)
        logdir_val = self._get_param_value("logdir")
        if logdir_val:
            cmd.extend(["--logdir", logdir_val])
        else:
            cmd.extend(["--logdir", str(PROJECT_ROOT / "perplexity_logs")])

        for group_name, params in PARAM_GROUPS:
            for key, _label, ptype, _default, _hint in params:
                val = self._get_param_value(key)
                # Skip logdir: always injected above
                if key == "logdir":
                    continue
                if val == _default or val == "" or val is None:
                    continue

                # Map key to CLI args
                cli_args = self._key_to_cli(key, val, ptype)
                if cli_args:
                    cmd.extend(cli_args)

        return cmd

    def _key_to_cli(self, key, val, ptype):
        """Convert a parameter key+value to CLI argument list."""
        mapping = {
            "model":             ["-m", str(val)],
            "prompt_file":       ["-f", str(val)],
            "binary_file":       ["-bf", str(val)],
            "prompt":            ["-p", str(val)],
            "seed":              ["-s", str(val)],
            "n_threads":         ["-t", str(val)],
            "n_threads_batch":   ["-tb", str(val)],
            "n_batch":           ["-b", str(val)],
            "n_ubatch":          ["-ub", str(val)],
            "n_ctx":             ["-c", str(val)],
            "n_keep":            ["--keep", str(val)],
            "n_chunks":          ["--chunks", str(val)],
            "n_parallel":        ["--parallel", str(val)],
            "n_gpu_layers":      ["-ngl", str(val)],
            "main_gpu":          ["-mg", str(val)],
            "max_gpu_per_split": ["-mgs", str(val)],
            "tensor_split":      ["-ts", str(val)],
            "n_sequences":       ["-ns", str(val)],
            "cont_batching":     [] if val else ["-nocb"],
            "ctx_shift":         [] if val else ["--no-context-shift"],
            "devices":           ["--device", str(val)] if val else [],
            "no_kv_offload":     ["--no-kv-offload"],
            "flash_attn":        (["-fa"] if val else ["-no-fa"]),
            "mla_attn":          ["-mla", str(val)],
            "attn_max_batch":    ["-amb", str(val)],
            "fused_moe_up_gate": ([] if val else ["-no-fmoe"]),
            "fused_up_gate":     ([] if val else ["-no-fug"]),
            "fused_mmad":        ([] if val else ["-no-mmad"]),
            "graph_reuse":       ["-gr"] if val else ["-no-gr"],
            "grouped_expert_routing": ["-ger"] if val else [],
            "merge_qkv":         ["-mqkv"] if val else [],
            "merge_up_gate_exps":["-muge"] if val else [],
            "k_cache_hadamard":  ["-khad"] if val else [],
            "v_cache_hadamard":  ["-vhad"] if val else [],
            "rope_cache":        ["-rcache"] if val else [],
            "cache_type_k":      ["-ctk", str(val)],
            "cache_type_v":      ["-ctv", str(val)],
            "reduce_type":       ["-grt", str(val)],
            "graph_attn_precision": ["-gap", str(val)] if val and val != "f16" else [],
            "has_mtp":           ["--mtp"] if val else [],
            "fused_delta_net":   ["--fused-delta-net", str(val)] if val else [],
            "dump_kv_cache":     ["-dkvc"] if val else [],
            "split_adjust_step_frequency": ["-sasf", str(val)] if val != 0.5 else [],
            "worst_graph_tokens": ["-wgt", str(val)] if val else [],
            "ncmoe":             ["-cmoe", str(val)] if val else [],
            "fit":               ["--fit"] if val else [],
            "fit_margin":        ["--fit-margin", str(val)] if val else [],
            "indexer_cache_type_k": ["-ictk", str(val)],
            "dry_run":           ["-dr"] if val else [],
            "dsa":               ["--dsa"] if val else [],
            "dsa_top_k":         ["--dsa-top-k", str(val)],
            "ppl_stride":        ["--ppl-stride", str(val)],
            "ppl_output_type":   ["--ppl-output-type", str(val)],
            "hellaswag":         ["--hellaswag"] if val else [],
            "hellaswag_tasks":   ["--hellaswag-tasks", str(val)],
            "winogrande":        ["--winogrande"] if val else [],
            "winogrande_tasks":  ["--winogrande-tasks", str(val)],
            "multiple_choice":   ["--multiple-choice"] if val else [],
            "multiple_choice_tasks": ["--multiple-choice-tasks", str(val)],
            "kl_divergence":     ["--kl-divergence"] if val else [],
            "logdir":            ["--logdir", str(val)],
            "logits_file":       ["--logits-file", str(val)],
            "model_alias":       ["--alias", str(val)],
            "split_mode":        ["-sm", str(val)],
            "split_output_tensor": ["-sot"] if val else [],
            "split_mode_tensor_parallel_scheduling": ["-smtps"] if val else [],
            "scheduler_async":   ["-sas"] if val else [],
            "rope_freq_base":    ["--rope-freq-base", str(val)],
            "rope_freq_scale":   ["--rope-freq-scale", str(val)],
            "yarn_ext_factor":   ["--yarn-ext-factor", str(val)],
            "yarn_attn_factor":  ["--yarn-attn-factor", str(val)],
            "yarn_beta_fast":    ["--yarn-beta-fast", str(val)],
            "yarn_beta_slow":    ["--yarn-beta-slow", str(val)],
            "yarn_orig_ctx":     ["--yarn-orig-ctx", str(val)],
            "defrag_thold":      ["--defrag-thold", str(val)],
            "grp_attn_n":        ["--grp-attn-n", str(val)],
            "grp_attn_w":        ["--grp-attn-w", str(val)],
            "numa":              ["--numa", str(val)] if val != "disabled" else [],
            "sched_max_copies":  ["-smc", str(val)],
            "n_print":           ["-ptc", str(val)],
            "ignore_unknown":    ["-iu"] if val else [],
            "ignore_eos":        ["--ignore-eos"] if val else [],
            "escape":            [] if val else ["--no-escape"],
            "use_color":         ["-co"] if val else [],
            "verbose_prompt":    ["--verbose-prompt"] if val else [],
            "warmup":            [] if val else ["--no-warmup"],
            "batch_warmup":      ["--warmup-batch"] if val else [],
            "use_mmap":          [] if val else ["--no-mmap"],
            "use_mlock":         ["--mlock"] if val else [],
            "check_tensors":     ["--check-tensors"] if val else [],
            "repack_tensors":    ["--repack-tensors"] if val else [],
            "validate_quants":   ["-vq"] if val else [],
            "only_active_exps":  [] if val else ["-no-ooae"],
            "defer_experts":     ["--defer-experts"] if val else [],
            "min_experts":       ["-ser", f"{val},0"] if isinstance(val, int) and val >= 0 else [],
            "pipeline":          ["-pipe", "lookahead" if val == 1 else ("selfcopy" if val == 2 else "0")],
            "lora_adapters":     self._multi_file_args("--lora", val),
            "control_vectors":   self._multi_file_args("--control-vector", val),
            "in_files":          self._multi_file_args("--in-file", val),
            "rpc_servers":       ["--rpc", str(val)],
            "cuda_params":       ["-cuda", str(val)] if val else [],

            # --tinylog and --minilog
            "tinylog":           ["--tinylog"] if val else [],
        }
        return mapping.get(key)

    def _multi_file_args(self, flag, val):
        if not val or val == "":
            return []
        parts = [x.strip() for x in str(val).split(";") if x.strip()]
        args = []
        for p in parts:
            args.extend([flag, p])
        return args

    def _log(self, text, tag=None):
        tag = self._classify_line(text, tag)
        if tag:
            self._output_text.insert("end", text, tag)
        else:
            self._output_text.insert("end", text)
        self._output_text.see("end")
        self.root.update_idletasks()

    def _classify_line(self, text, hint_tag):
        if hint_tag:
            return hint_tag
        t = text.strip()
        if t.startswith("CMD:"):
            return "cmd"
        if t.startswith("ERROR") or t.startswith("FAIL") or "error:" in t.lower():
            return "error"
        if "perplexity" in t.lower() or "ppl" in t.lower() or (
            re.search(r'\[[\d,]+\][\d.]+', t) and ('Final' not in t)):
            return "ppl"
        if t.startswith("--") or t.startswith("==="):
            return "info"
        if "Final" in t or "result" in t.lower() or "score" in t.lower():
            return "result"
        if t.startswith("[") and t.rstrip().endswith(","):
            return "ppl"
        return "stdout"

    def _read_stream(self, stream, prefix=""):
        """Read a process stdout stream in large chunks, emit to log in real-time.
        Returns (last_lines, full_text)."""
        last_lines = []
        pending = ""
        CHUNK = 65536
        while True:
            raw = stream.read(CHUNK)
            if not raw:
                break
            text = raw.decode("utf-8", errors="replace")
            pending += text
            # process complete lines
            while True:
                nl_idx = -1
                for nl in ("\n", "\r"):
                    idx = pending.find(nl)
                    if idx != -1 and (nl_idx == -1 or idx < nl_idx):
                        nl_idx = idx
                if nl_idx == -1:
                    break
                line_text = pending[:nl_idx + 1]
                self.root.after(0, self._log, f"{prefix}{line_text}")
                clean = pending[:nl_idx].rstrip("\r\n")
                if clean:
                    last_lines.append(clean)
                    if len(last_lines) > 2:
                        last_lines.pop(0)
                pending = pending[nl_idx + 1:]
            # flush partial if 32+ chars accumulated
            if len(pending) >= 32:
                self.root.after(0, self._log, f"{prefix}{pending}")
                pending = ""
        if pending:
            self.root.after(0, self._log, f"{prefix}{pending}")
            clean = pending.rstrip("\r\n")
            if clean:
                last_lines.append(clean)
                if len(last_lines) > 2:
                    last_lines.pop(0)
        return last_lines, ""

    def _clear_output(self):
        self._output_text.delete("1.0", "end")

    def _launch(self):
        cmd = self._build_cmd()
        if cmd is None:
            return
        self._launch_btn.config(state="disabled")
        self.status_var.set("Running...")
        self._log(f"CMD: {' '.join(str(c) for c in cmd)}\n", "cmd")
        self._log("--- Launching ---\n", "info")
        threading.Thread(target=self._run_process, args=(cmd,), daemon=True).start()

    def _run_process(self, cmd):
        try:
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=0,
                cwd=str(PROJECT_ROOT),
            )
            last_lines, _ = self._read_stream(self.process.stdout)
            self.process.wait()
            self.root.after(0, self._on_process_done, self.process.returncode, last_lines)
        except Exception as e:
            self.root.after(0, self._log, f"ERROR: {e}\n", "error")
            self.root.after(0, self._on_process_done, -1, [])

    def _on_process_done(self, returncode, last_lines):
        self.process = None
        self._launch_btn.config(state="normal")
        self.status_var.set("Done" if returncode == 0 else f"Failed ({returncode})")
        self._log(f"--- Process exit code: {returncode} ---\n", "info")
        # Extract PPL from last lines
        ppl_line = self._extract_ppl(last_lines)
        if ppl_line:
            self._result_var.set(ppl_line)
            self._save_result(ppl_line)

    def _extract_ppl(self, lines):
        for line in reversed(lines):
            m = re.search(r"perplexity:\s+([\d.]+)", line, re.IGNORECASE)
            if m:
                return f"PPL: {m.group(1)}"
            m = re.search(r"Final estimate:\s+PPL.*?([\d.]+)", line)
            if m:
                return f"PPL: {m.group(1)}"
            m = re.search(r"Final result:\s+([\d.]+)", line)
            if m:
                return f"Score: {m.group(1)}"
        return ""

    def _stop_process(self):
        if self.process and self.process.poll() is None:
            self.process.terminate()
            self._log("--- Process terminated ---\n", "error")
            self.status_var.set("Terminated")

    # ------------------------------------------------------------------
    # Chain launch
    # ------------------------------------------------------------------

    def _launch_chain(self):
        if not self.chain_list:
            messagebox.showinfo("Chain", "No profiles in chain.", parent=self.root)
            return
        self._chain_launch_btn.config(state="disabled")
        self._log("=== Chain launch started ===\n", "info")
        threading.Thread(target=self._run_chain, daemon=True).start()

    def _run_chain(self):
        for idx, profile_path in enumerate(self.chain_list):
            if not os.path.isfile(profile_path):
                self.root.after(0, self._log, f"Chain #{idx+1}: file not found {profile_path}\n", "error")
                continue
            self.root.after(0, self._log, f"=== Chain #{idx+1}: {os.path.basename(profile_path)} ===\n", "info")
            with open(profile_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.root.after(0, self._apply_params, data)
            # Small delay to let UI update, then run
            time.sleep(0.3)
            cmd = self._build_cmd()
            if cmd is None:
                continue
            self.root.after(0, self._log, f"CMD: {' '.join(str(c) for c in cmd)}\n", "cmd")
            try:
                proc = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    bufsize=0,
                    cwd=str(PROJECT_ROOT),
                )
                last_lines, _ = self._read_stream(proc.stdout, prefix=f"[#{idx+1}] ")
                proc.wait()
                ppl_line = self._extract_ppl(last_lines)
                if ppl_line:
                    self.root.after(0, self._result_var.set, ppl_line)
                    self.root.after(0, self._save_result, ppl_line)
                self.root.after(0, self._log, f"=== Chain #{idx+1} exit code: {proc.returncode} ===\n", "info")
            except Exception as e:
                self.root.after(0, self._log, f"Chain #{idx+1} ERROR: {e}\n", "error")
        self.root.after(0, lambda: self._chain_launch_btn.config(state="normal"))
        self.root.after(0, self._log, "=== Chain launch completed ===\n", "info")
        self.root.after(0, self.status_var.set, "Chain done")

    # ------------------------------------------------------------------
    # Results
    # ------------------------------------------------------------------

    def _save_result(self, text):
        try:
            if RESULTS_FILE.exists():
                with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                    results = json.load(f)
            else:
                results = []
            results.append({
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "result": text,
                "profile": self._profile_var.get(),
                "model": self._get_param_value("model"),
            })
            with open(RESULTS_FILE, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        except Exception as e:
            self._log(f"Results save error: {e}\n", "error")

    def _load_results(self):
        if RESULTS_FILE.exists():
            try:
                with open(RESULTS_FILE, "r", encoding="utf-8") as f:
                    results = json.load(f)
                if results:
                    last = results[-1]
                    self._result_var.set(last.get("result", ""))
            except Exception:
                pass

    def _view_results(self):
        if not RESULTS_FILE.exists():
            messagebox.showinfo("Results", "No results yet.", parent=self.root)
            return
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            results = json.load(f)
        text = json.dumps(results, indent=2)
        win = tk.Toplevel(self.root)
        win.title("Results")
        win.geometry("600x400")
        t = tk.Text(win, wrap="none")
        s = ttk.Scrollbar(win, orient="vertical", command=t.yview)
        t.configure(yscrollcommand=s.set)
        t.pack(side="left", fill="both", expand=True)
        s.pack(side="right", fill="y")
        t.insert("1.0", text)
        t.configure(state="disabled")


def main():
    root = tk.Tk()
    app = PerplexityGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
