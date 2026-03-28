#include "llama-mmap.h"

#include "llama-impl.h"

#include "ggml.h"

#include <cstring>
#include <climits>
#include <stdexcept>
#include <cerrno>
#include <algorithm>
#include <fstream>
#include <sstream>

#ifdef __has_include
    #if __has_include(<unistd.h>)
        #include <unistd.h>
        #include <fcntl.h>
        #include <sys/stat.h>
        #if defined(_POSIX_MAPPED_FILES)
            #include <sys/mman.h>
        #endif
        #if defined(_POSIX_MEMLOCK_RANGE)
            #include <sys/resource.h>
        #endif
    #endif
#endif

#if defined(_WIN32)
    #define WIN32_LEAN_AND_MEAN
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #ifndef PATH_MAX
        #define PATH_MAX MAX_PATH
    #endif
    #include <io.h>
#endif

#if defined(__APPLE__)
#include <TargetConditionals.h>
#endif

// TODO: consider moving to llama-impl.h if needed in more places
#if defined(_WIN32)
static std::string llama_format_win_err(DWORD err) {
    LPSTR buf;
    size_t size = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                 NULL, err, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&buf, 0, NULL);
    if (!size) {
        return "FormatMessageA failed";
    }
    std::string ret(buf, size);
    LocalFree(buf);
    return ret;
}
#endif

// llama_file

struct llama_file::impl {
#if defined(_WIN32)
    HANDLE fp_win32;
    DWORD file_flags;
    bool use_no_buffering;
    std::string name;

    std::string GetErrorMessageWin32(DWORD error_code) const {
        std::string ret;
        LPSTR lpMsgBuf = NULL;
        DWORD bufLen = FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM | FORMAT_MESSAGE_IGNORE_INSERTS,
                                    NULL, error_code, MAKELANGID(LANG_NEUTRAL, SUBLANG_DEFAULT), (LPSTR)&lpMsgBuf, 0, NULL);
        if (!bufLen) {
            ret = format("Win32 error code: %lx", error_code);
        } else {
            ret = lpMsgBuf;
            LocalFree(lpMsgBuf);
        }

        return ret;
    }

    size_t read_direct(void * ptr, size_t len, size_t offset) const {
        SYSTEM_INFO siSysInfo;
        GetSystemInfo(&siSysInfo);
        DWORD dwPageSize = siSysInfo.dwPageSize;
        GGML_ASSERT((uintptr_t) ptr % dwPageSize == 0);
        GGML_ASSERT(len % dwPageSize == 0);
        GGML_ASSERT(offset % dwPageSize == 0);

        HANDLE hFile = ReOpenFile((HANDLE) _get_osfhandle(_fileno(fp)), GENERIC_READ, FILE_SHARE_READ, FILE_FLAG_NO_BUFFERING);
        if (hFile == INVALID_HANDLE_VALUE) {
            throw std::runtime_error(format("failed to open %s: %s", name.c_str(), llama_format_win_err(GetLastError()).c_str()));
        }

        size_t bytes_read = 0;
        while (len > 0) {
            OVERLAPPED oOverlap = {};
            oOverlap.OffsetHigh = offset >> 32;
            oOverlap.Offset = offset;
            DWORD nBytesToRead = std::min(len, (size_t) 0xFFFFFFFF & ~(dwPageSize - 1));
            DWORD count = 0;
            if (!ReadFile(hFile, ptr, nBytesToRead, &count, &oOverlap)) {
                if (GetLastError() == ERROR_HANDLE_EOF) {
                    bytes_read += count;
                    break;
                }
                throw std::runtime_error(format("direct read error: %s", llama_format_win_err(GetLastError()).c_str()));
            }
            bytes_read += count;
            if (count < nBytesToRead) {
                break;
            }
            ptr = (char *) ptr + count;
            offset += count;
            len -= count;
        }

        CloseHandle(hFile);

        return bytes_read;
    }

    static constexpr bool DIRECT_IO_SUPPORTED = true;

    impl(const char * fname, const char * mode, const bool use_direct_io = false) : use_no_buffering(false), name(fname) {
        file_flags = FILE_ATTRIBUTE_NORMAL;
        fp = ggml_fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }

        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            std::fclose(fp);
            fp = NULL;

            DWORD desired_access = GENERIC_READ;
            DWORD creation_disposition = OPEN_EXISTING;
            DWORD share_mode = FILE_SHARE_READ;

            fp_win32 = CreateFileA(fname, desired_access, share_mode, NULL, creation_disposition,
                                    FILE_FLAG_SEQUENTIAL_SCAN, NULL);

            if (fp_win32 == INVALID_HANDLE_VALUE) {
                DWORD err = GetLastError();
                LLAMA_LOG_WARN("Failed to open '%s' with FILE_FLAG_SEQUENTIAL_SCAN (error %lu). Falling back to buffered I/O.\n",
                               fname, err);
                fp = ggml_fopen(fname, mode);
                if (fp == NULL) {
                    throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
                }
                fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
            } else {
                use_no_buffering = true;

                LARGE_INTEGER li;
                li.QuadPart = 0;
                if (!GetFileSizeEx(fp_win32, &li)) {
                    CloseHandle(fp_win32);
                    throw std::runtime_error(format("failed to get file size: %s", GetErrorMessageWin32(GetLastError()).c_str()));
                }
                size = li.QuadPart;

                alignment = 4096;

                LARGE_INTEGER zero = {0};
                SetFilePointerEx(fp_win32, zero, NULL, FILE_BEGIN);
                return;
            }
        } else {
            fp_win32 = (HANDLE) _get_osfhandle(_fileno(fp));
        }

        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        LARGE_INTEGER li;
        li.QuadPart = 0;
        BOOL ret = SetFilePointerEx(fp_win32, li, &li, FILE_CURRENT);
        if (!ret) {
            throw std::runtime_error(format("tell error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }

        return li.QuadPart;
    }

    void seek(size_t offset, int whence) const {
        static_assert(SEEK_SET == FILE_BEGIN, "SEEK_SET != FILE_BEGIN");
        static_assert(SEEK_CUR == FILE_CURRENT, "SEEK_CUR != FILE_CURRENT");
        static_assert(SEEK_END == FILE_END, "SEEK_END != FILE_END");

        LARGE_INTEGER li;
        li.QuadPart = offset;
        BOOL ret = SetFilePointerEx(fp_win32, li, NULL, whence);
        if (!ret) {
            throw std::runtime_error(format("seek error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (len == 0) return;
        size_t bytes_read = 0;
        while (bytes_read < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_read, 64*1024*1024);
            DWORD chunk_read = 0;
            BOOL result = ReadFile(fp_win32, reinterpret_cast<char*>(ptr) + bytes_read, chunk_size, &chunk_read, NULL);
            if (!result) {
                throw std::runtime_error(format("read error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_read < chunk_size || chunk_read == 0) {
                throw std::runtime_error("unexpectedly reached end of file");
            }

            bytes_read += chunk_read;
        }
    }

    void read_aligned_chunk(void * dest, size_t read_size) {
        size_t offset = tell();
        size_t aligned_offset = offset & ~(alignment - 1);
        size_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + read_size + alignment - 1) & ~(alignment - 1);

        static thread_local struct {
            void*  ptr;
            size_t buf_size;
            size_t buf_align;
        } cache = {nullptr, 0, 0};

        if (!cache.ptr || cache.buf_size < bytes_to_read || cache.buf_align < alignment) {
            if (cache.ptr) VirtualFree(cache.ptr, 0, MEM_RELEASE);
            cache.ptr = VirtualAlloc(NULL, bytes_to_read, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE);
            if (!cache.ptr) {
                throw std::runtime_error(format("VirtualAlloc failed for aligned buffer"));
            }
            cache.buf_size = bytes_to_read;
            cache.buf_align = alignment;
        }

        seek(aligned_offset, SEEK_SET);
        read_raw_unsafe(cache.ptr, bytes_to_read);

        memcpy(dest, static_cast<char*>(cache.ptr) + offset_from_alignment, read_size);
    }

    void read_raw(void * ptr, size_t len) {
        read_raw_unsafe(ptr, len);
    }

    uint32_t read_u32() {
        uint32_t val;
        read_raw(&val, sizeof(val));
        return val;
    }

    void write_raw(const void * ptr, size_t len) const {
        size_t bytes_written = 0;
        while (bytes_written < len) {
            size_t chunk_size = std::min<size_t>(len - bytes_written, 64*1024*1024);
            DWORD chunk_written = 0;
            BOOL result = WriteFile(fp_win32, reinterpret_cast<char const*>(ptr) + bytes_written, chunk_size, &chunk_written, NULL);
            if (!result) {
                throw std::runtime_error(format("write error: %s", GetErrorMessageWin32(GetLastError()).c_str()));
            }
            if (chunk_written < chunk_size || chunk_written == 0) {
                throw std::runtime_error("unexpectedly failed to write bytes");
            }

            bytes_written += chunk_written;
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return use_no_buffering && alignment > 1;
    }

    ~impl() {
        if (use_no_buffering && fp_win32 != INVALID_HANDLE_VALUE) {
            CloseHandle(fp_win32);
        } else if (fp) {
            std::fclose(fp);
        }
    }
#else
    impl(const char * fname, const char * mode, [[maybe_unused]] const bool use_direct_io = false) : fname(fname) {
#ifdef __linux__
        // Try unbuffered I/O for read only
        if (use_direct_io && std::strcmp(mode, "rb") == 0) {
            if (init_fd()) {
                return;
            }
            LLAMA_LOG_WARN("Failed to open file '%s' with error: %s. Falling back to buffered I/O",
                           fname, strerror(errno));
        }
#endif
        init_fp(mode);
    }

#ifdef __linux__
    bool init_fd() {
        fd = open(fname.c_str(), O_RDONLY | O_DIRECT);

        if (fd != -1) {
            struct stat file_stats{};
            fstat(fd, &file_stats);

            size = file_stats.st_size;
            alignment = file_stats.st_blksize;

            off_t ret = lseek(fd, 0, SEEK_SET);
            if (ret == -1) {
                throw std::runtime_error(format("seek error: %s", strerror(errno)));
            }
            return true;
        }
        return false;
    }
#endif

    void init_fp(const char * mode) {
        fp = ggml_fopen(fname.c_str(), mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        seek(0, SEEK_END);
        size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        if (fd == -1) {
            long ret = std::ftell(fp);
            if (ret == -1) {
                throw std::runtime_error(format("ftell error: %s", strerror(errno)));
            }

            return (size_t) ret;
        }

        off_t pos = lseek(fd, 0, SEEK_CUR);
        if (pos == -1) {
            throw std::runtime_error(format("lseek error: %s", strerror(errno)));
        }
        return (size_t) pos;
    }

    void seek(size_t offset, int whence) const {
        off_t ret = 0;
        if (fd == -1) {
            ret = std::fseek(fp, (long) offset, whence);
        } else {
            ret = lseek(fd, offset, whence);
        }
        if (ret == -1) {
            throw std::runtime_error(format("seek error: %s", strerror(errno)));
        }
    }

    void read_raw_unsafe(void * ptr, size_t len) {
        if (len == 0) {
            return;
        }
        errno = 0;
        if (fd == -1) {
            const size_t curr_off = tell();
            const size_t to_read = std::min(len, size - curr_off);

            std::size_t ret = std::fread(ptr, to_read, 1, fp);
            if (ferror(fp)) {
                throw std::runtime_error(format("read error: %s", strerror(errno)));
            }
            if (to_read > 0 && ret != 1) {
                throw std::runtime_error("unexpectedly reached end of file");
            }
        } else {
            size_t bytes_read = 0;
            while (bytes_read < len) {
                const size_t to_read = len - bytes_read;
                ssize_t ret = ::read(fd, reinterpret_cast<char *>(ptr) + bytes_read, to_read);

                if (ret == -1) {
                    if (errno == EINTR) {
                        continue;  // Interrupted by signal, retry
                    }
                    // Fallback to std::fread in case the DMA controller cannot access the buffer
                    if (errno == EFAULT || errno == EINVAL) {
                        LLAMA_LOG_WARN("%s: Falling back to buffered IO due to %s\n", __func__, strerror(errno));
                        auto curr_off = tell();
                        close(fd);
                        fd = -1;
                        alignment = 1;
                        init_fp("rb");
                        seek(curr_off, SEEK_SET);
                        read_raw_unsafe(ptr, len);
                        return;
                    }
                    throw std::runtime_error(format("read error: %s", strerror(errno)));
                }
                if (ret == 0) {
                    // EOF: allow if this read was only pulling alignment padding past file end
                    off_t pos = lseek(fd, 0, SEEK_CUR);
                    if (pos != -1 && (size_t) pos == size) {
                        std::memset(reinterpret_cast<char *>(ptr) + bytes_read, 0, len - bytes_read);
                        return;
                    }
                    throw std::runtime_error("unexpectedly reached end of file");
                }

                bytes_read += (size_t) ret;
            }
        }
    }

    void read_aligned_chunk(void * dest, size_t size) {
        size_t offset = tell();
        off_t aligned_offset = offset & ~(alignment - 1);
        off_t offset_from_alignment = offset - aligned_offset;
        size_t bytes_to_read = (offset_from_alignment + size + alignment - 1) & ~(alignment - 1);

        // MONOTONIC CACHE: Stores the buffer and the alignment it satisfies
        static thread_local struct {
            void*  ptr = nullptr;
            size_t buf_size = 0;
            size_t buf_align = 0;
        } cache;

        if (!cache.ptr || cache.buf_size < bytes_to_read || cache.buf_align < alignment) {
            // Free old buffer if exists
            if (cache.ptr) ::free(cache.ptr);
            //~ LLAMA_LOG_INFO("%s: using posix_memalign cache\n", __func__);
            void* raw = nullptr;
            int ret = posix_memalign(&raw, alignment, bytes_to_read);
            if (ret != 0) {
                throw std::runtime_error(format("posix_memalign failed with error %d", ret));
            }
            cache.ptr = raw;
            cache.buf_size = bytes_to_read;
            cache.buf_align = alignment; // Remember what we got
        }

        // Use the cached buffer (already aligned to at least `alignment`)
        seek(aligned_offset, SEEK_SET);
        read_raw_unsafe(cache.ptr, bytes_to_read);
        //~ LLAMA_LOG_INFO("%s: use the cached buffer\n", __func__);

        memcpy(dest, static_cast<char*>(cache.ptr) + offset_from_alignment, size);
    }

    void read_raw(void * ptr, size_t len) {
        if (len == 0) return;
        if (has_direct_io()) {
            read_aligned_chunk(ptr, len);
        } else {
            read_raw_unsafe(ptr, len);
        }
    }

    uint32_t read_u32() {
        uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    void write_raw(const void * ptr, size_t len) const {
        if (len == 0) {
            return;
        }
        errno = 0;
        size_t ret = std::fwrite(ptr, len, 1, fp);
        if (ret != 1) {
            throw std::runtime_error(format("write error: %s", strerror(errno)));
        }
    }

    void write_u32(uint32_t val) const {
        write_raw(&val, sizeof(val));
    }

    bool has_direct_io() const {
        return fd != -1 && alignment > 1;
    }

#if defined(__linux__) || defined(__FreeBSD__) || defined(__APPLE__)
    size_t read_direct(void * ptr, size_t len, size_t offset) const {
        int page_size = sysconf(_SC_PAGESIZE);
        GGML_ASSERT((uintptr_t) ptr % page_size == 0);
        GGML_ASSERT(len % page_size == 0);
        GGML_ASSERT(offset % page_size == 0);
#ifdef __APPLE__
        int fddl = open(fname.c_str(), O_RDONLY);
        if (fddl == -1) {
            throw std::runtime_error(format("failed to open %s: %s", fname.c_str(), strerror(errno)));
        }
        if (fcntl(fddl, F_NOCACHE, 1) == -1) {
            throw std::runtime_error(format("failed to enable direct I/O: %s", strerror(errno)));
        }
#else
        int fddl = open(fname.c_str(), O_RDONLY | O_DIRECT);
        if (fddl == -1) {
            throw std::runtime_error(format("failed to open %s for direct I/O: %s", fname.c_str(), strerror(errno)));
        }
#endif
        size_t bytes_read = 0;
        while (len > 0) {
            ssize_t count = pread(fddl, ptr, std::min(len, (size_t) INT_MAX & ~(page_size - 1)), offset);
            if (count == -1) {
                throw std::runtime_error(format("direct read error: %s", strerror(errno)));
            }
            if (count == 0) {
                break;
            }
            ptr = (char *) ptr + count;
            offset += count;
            len -= count;
            bytes_read += count;
        }

        close(fddl);

        return bytes_read;
    }

    static constexpr bool DIRECT_IO_SUPPORTED = true;
#else
    static constexpr bool DIRECT_IO_SUPPORTED = false;
#endif

    ~impl() {
        if (fd != -1) {
            close(fd);
        } else {
            std::fclose(fp);
        }
    }
    int fd = -1;
    std::string fname;
#endif

    size_t read_alignment() const {
        return alignment;
    }

    size_t alignment = 1;

    FILE * fp{};
    size_t size{};
};

llama_file::llama_file(const char * fname, const char * mode, const bool use_direct_io) :
    pimpl(std::make_unique<impl>(fname, mode, use_direct_io)) {}
llama_file::~llama_file() = default;

size_t llama_file::tell() const { return pimpl->tell(); }
size_t llama_file::size() const { return pimpl->size; }

size_t llama_file::read_alignment() const { return pimpl->read_alignment(); }
bool llama_file::has_direct_io() const { return pimpl->has_direct_io(); }

int llama_file::file_id() const {
#ifdef _WIN32
    return _fileno(pimpl->fp);
#else
    if (pimpl->fd != -1) {
        return pimpl->fd;
    }
#if defined(fileno)
    return fileno(pimpl->fp);
#else
    return ::fileno(pimpl->fp);
#endif
#endif
}

void llama_file::seek(size_t offset, int whence) const { pimpl->seek(offset, whence); }
void llama_file::read_raw(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }
#ifdef _WIN32
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw(ptr, len); }
#else
void llama_file::read_raw_unsafe(void * ptr, size_t len) { pimpl->read_raw_unsafe(ptr, len); }
#endif
void llama_file::read_aligned_chunk(void * dest, size_t size) { pimpl->read_aligned_chunk(dest, size); }
size_t llama_file::read_direct(void * ptr, size_t len, size_t offset) const { return pimpl->read_direct(ptr, len, offset); }

uint32_t llama_file::read_u32() { return pimpl->read_u32(); }

void llama_file::write_raw(const void * ptr, size_t len) const { pimpl->write_raw(ptr, len); }
void llama_file::write_u32(uint32_t val) const { pimpl->write_u32(val); }

// llama_mmap

struct llama_mmap::impl {
#ifdef _POSIX_MAPPED_FILES
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    impl(struct llama_file * file, size_t prefetch, bool numa, [[maybe_unused]] bool use_thp) {
        size = file->size();
        int fd = file->file_id();
        int flags = MAP_SHARED;
        if (numa) { prefetch = 0; }
#ifdef __linux__
        if (posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL)) {
            LLAMA_LOG_WARN("warning: posix_fadvise(.., POSIX_FADV_SEQUENTIAL) failed: %s\n",
                    strerror(errno));
        }
        if (prefetch) { flags |= MAP_POPULATE; }
        if (use_thp) {
            size_t huge = get_default_huge_page_size();
            auto size = huge*((file->size() + huge - 1)/huge);
            addr = mmap(nullptr, size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
            if (addr != MAP_FAILED) {
                printf("%s: using THP with page size %zu MiB ", __func__, huge/(1024*1024));
                fflush(stdout);
                size_t tot = 0;
                while (tot < file->size()) {
                    auto n_read = pread(fd, static_cast<char*>(addr) + tot, file->size() - tot, tot);
                    if (n_read < 0) throw std::runtime_error(format("Reading into mapped huge pages failed at %zu (%s)", tot, strerror(errno)));
                    printf(".");  fflush(stdout);
                    tot += n_read;
                }
                printf(" done\n");
                mapped_fragments.emplace_back(0, file->size());
                mapped_page_size = huge;
                return;
            }
            else {
                fprintf(stderr, "%s: mmap with huge page size %zu MiB failed (%s)\n", __func__, huge/(1024*1024), strerror(errno));
            }
        }
#endif
        addr = mmap(NULL, file->size(), PROT_READ, flags, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        if (prefetch > 0) {
            if (posix_madvise(addr, std::min(file->size(), prefetch), POSIX_MADV_WILLNEED)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_WILLNEED) failed: %s\n",
                        strerror(errno));
            }
        }
        if (numa) {
            if (posix_madvise(addr, file->size(), POSIX_MADV_RANDOM)) {
                LLAMA_LOG_WARN("warning: posix_madvise(.., POSIX_MADV_RANDOM) failed: %s\n",
                        strerror(errno));
            }
        }

        mapped_fragments.emplace_back(0, file->size());
    }

#ifdef __linux__
    static int get_default_huge_page_size() {
        int pg_size = 2048;
        std::ifstream in("/proc/meminfo");
        if (in) {
            std::string line;
            while (true) {
                std::getline(in, line);
                if (in.fail()) break;
                if (auto pos = line.find("Hugepagesize:"); pos != std::string::npos) {
                    std::istringstream str(line.data() + pos + 13);
                    int aux;
                    str >> aux;
                    if (!str.fail()) pg_size = aux;
                    break;
                }
            }
        }
        return pg_size * 1024;
    }
#endif


    static void align_range(size_t * first, size_t * last, size_t page_size) {
        size_t offset_in_page = *first & (page_size - 1);
        size_t offset_to_page = offset_in_page == 0 ? 0 : page_size - offset_in_page;
        *first += offset_to_page;

        *last = *last & ~(page_size - 1);

        if (*last <= *first) {
            *last = *first;
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        int page_size = mapped_page_size > 0 ? mapped_page_size : sysconf(_SC_PAGESIZE);
        align_range(&first, &last, page_size);
        size_t len = last - first;

        if (len == 0) {
            return;
        }

        GGML_ASSERT(first % page_size == 0);
        GGML_ASSERT(last % page_size == 0);
        GGML_ASSERT(last > first);

        void * next_page_start = (uint8_t *) addr + first;

        if (munmap(next_page_start, len)) {
            LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
        }

        std::vector<std::pair<size_t, size_t>> new_mapped_fragments;
        for (const auto & frag : mapped_fragments) {
            if (frag.first < first && frag.second > last) {
                new_mapped_fragments.emplace_back(frag.first, first);
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first < first && frag.second > first) {
                new_mapped_fragments.emplace_back(frag.first, first);
            } else if (frag.first < last && frag.second > last) {
                new_mapped_fragments.emplace_back(last, frag.second);
            } else if (frag.first >= first && frag.second <= last) {
            } else {
                new_mapped_fragments.push_back(frag);
            }
        }
        mapped_fragments = std::move(new_mapped_fragments);
    }

    ~impl() {
        for (const auto & frag : mapped_fragments) {
            if (munmap((char *) addr + frag.first, frag.second - frag.first)) {
                LLAMA_LOG_WARN("warning: munmap failed: %s\n", strerror(errno));
            }
        }
    }
#elif defined(_WIN32)
    impl(struct llama_file * file, size_t prefetch, bool numa, [[maybe_unused]] bool use_thp) {
        GGML_UNUSED(numa);

        size = file->size();

        HANDLE hFile = (HANDLE) _get_osfhandle(file->file_id());

        HANDLE hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);

        if (hMapping == NULL) {
            DWORD error = GetLastError();
            throw std::runtime_error(format("CreateFileMappingA failed: %s", llama_format_win_err(error).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
        DWORD error = GetLastError();
        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(error).c_str()));
        }

        if (prefetch > 0) {
#if _WIN32_WINNT >= 0x602
            BOOL (WINAPI *pPrefetchVirtualMemory) (HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
            HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

            pPrefetchVirtualMemory = (decltype(pPrefetchVirtualMemory))(void *) GetProcAddress(hKernel32, "PrefetchVirtualMemory");

            if (pPrefetchVirtualMemory) {
                WIN32_MEMORY_RANGE_ENTRY range;
                range.VirtualAddress = addr;
                range.NumberOfBytes = (SIZE_T) std::min(size, prefetch);
                if (!pPrefetchVirtualMemory(GetCurrentProcess(), 1, &range, 0)) {
                    LLAMA_LOG_WARN("warning: PrefetchVirtualMemory failed: %s\n",
                            llama_format_win_err(GetLastError()).c_str());
                }
            }
#else
            LLAMA_LOG_DEBUG("skipping PrefetchVirtualMemory because _WIN32_WINNT < 0x602\n");
#endif
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

    ~impl() {
        if (!UnmapViewOfFile(addr)) {
            LLAMA_LOG_WARN("warning: UnmapViewOfFile failed: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    impl(struct llama_file * file, size_t prefetch, bool numa, [[maybe_unused]] bool use_thp) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);
        GGML_UNUSED(numa);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }
#endif

    void * addr;
    size_t size;
    size_t mapped_page_size = 0;
};

llama_mmap::llama_mmap(struct llama_file * file, size_t prefetch, bool numa, bool use_thp) :
    pimpl(std::make_unique<impl>(file, prefetch, numa, use_thp)), prefetch(prefetch > 0) {}
llama_mmap::~llama_mmap() = default;

size_t llama_mmap::size() const { return pimpl->size; }
void * llama_mmap::addr() const { return pimpl->addr; }

void llama_mmap::unmap_fragment(size_t first, size_t last) { pimpl->unmap_fragment(first, last); }

void llama_mmap::populate(size_t first, size_t last) const {
    GGML_UNUSED(first);
    GGML_UNUSED(last);
}

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mmap::SUPPORTED  = true;
#else
const bool llama_mmap::SUPPORTED  = false;
#endif

struct llama_anonymous_mmap {
    llama_file * file;
    void * addr;
    size_t size;
    bool prefetch;
    std::vector<std::pair<size_t, size_t>> mapped_fragments;

    llama_anonymous_mmap(const llama_anonymous_mmap &) = delete;

    size_t get_size() const { return size; }
    void * get_addr() const { return addr; }

#ifdef _POSIX_MAPPED_FILES
#ifndef MAP_ANONYMOUS
#define MAP_ANONYMOUS MAP_ANON
#endif
    llama_anonymous_mmap(struct llama_file * file, bool prefetch) : file(file), prefetch(prefetch) {
        size = file->size();
        addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_ANONYMOUS | MAP_PRIVATE, -1, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap(.., MAP_ANONYMOUS) failed: %s", strerror(errno)));
        }
#ifdef __linux__
        if (madvise(addr, size, MADV_HUGEPAGE)) {
            LLAMA_LOG_WARN("warning: madvise(.., MADV_HUGEPAGE) failed: %s\n", strerror(errno));
        }
#endif
        mapped_fragments.emplace_back(0, size);
    }

    void populate(size_t first, size_t last) const {
        int page_size = sysconf(_SC_PAGESIZE);

        size_t first_aligned = first & ~(page_size - 1);
        size_t last_aligned = (last + page_size - 1) & ~(page_size - 1);

        size_t bytes_read = file->read_direct((char *) addr + first_aligned, last_aligned - first_aligned, first_aligned);
        if (bytes_read != std::min(last_aligned, file->size) - first_aligned) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }

#elif defined(_WIN32)
    llama_anonymous_mmap(struct llama_file * file, bool prefetch) : file(file), prefetch(prefetch) {
        size = file->size();

        HANDLE hMapping = CreateFileMapping(INVALID_HANDLE_VALUE, NULL, PAGE_READWRITE, size >> 32, size, NULL);
        if (hMapping == NULL) {
            throw std::runtime_error(format("CreateFileMapping failed: %s", llama_format_win_err(GetLastError()).c_str()));
        }

        addr = MapViewOfFile(hMapping, FILE_MAP_ALL_ACCESS, 0, 0, size);
        DWORD dwError = GetLastError();

        CloseHandle(hMapping);

        if (addr == NULL) {
            throw std::runtime_error(format("MapViewOfFile failed: %s", llama_format_win_err(dwError).c_str()));
        }
    }

    void populate(size_t first, size_t last) const {
        SYSTEM_INFO siSysInfo;
        GetSystemInfo(&siSysInfo);
        DWORD dwPageSize = siSysInfo.dwPageSize;

        size_t first_aligned = first & ~(dwPageSize - 1);
        size_t last_aligned = (last + dwPageSize - 1) & ~(dwPageSize - 1);

        size_t bytes_read = file->read_direct((char *) addr + first_aligned, last_aligned - first_aligned, first_aligned);
        if (bytes_read != std::min<size_t>(last_aligned, file->size()) - first_aligned) {
            throw std::runtime_error("unexpectedly reached end of file");
        }
    }

    void unmap_fragment(size_t first, size_t last) {
        SYSTEM_INFO siSysInfo;
        GetSystemInfo(&siSysInfo);
        DWORD dwPageSize = siSysInfo.dwPageSize;

        size_t first_aligned = first & ~(dwPageSize - 1);
        size_t last_aligned = (last + dwPageSize - 1) & ~(dwPageSize - 1);

        DWORD (WINAPI *pOfferVirtualMemory) (PVOID, SIZE_T, DWORD);
        HMODULE hKernel32 = GetModuleHandleW(L"kernel32.dll");

        pOfferVirtualMemory = reinterpret_cast<decltype(pOfferVirtualMemory)> (GetProcAddress(hKernel32, "OfferVirtualMemory"));

        if (pOfferVirtualMemory) {
            if (pOfferVirtualMemory((char *) addr + first_aligned, last_aligned - first_aligned, 0x00000004)) {
                LLAMA_LOG_WARN("warning: OfferVirtualMemory failed: %s\n", llama_format_win_err(GetLastError()).c_str());
            }
        } else {
            LLAMA_LOG_WARN("warning: OfferVirtualMemory unavailable: %s\n", llama_format_win_err(GetLastError()).c_str());
            if (!VirtualAlloc((char *) addr + first_aligned, last_aligned - first_aligned, MEM_RESET, PAGE_READWRITE)) {
                LLAMA_LOG_WARN("warning: VirtualAlloc(.., MEM_RESET) failed: %s\n", llama_format_win_err(GetLastError()).c_str());
            }
        }
    }

#else
    llama_anonymous_mmap(struct llama_file * file, bool prefetch) {
        GGML_UNUSED(file);
        GGML_UNUSED(prefetch);

        throw std::runtime_error("mmap not supported");
    }

    void populate(size_t first, size_t last) const {
        GGML_UNUSED(first);
        GGML_UNUSED(last);

        throw std::runtime_error("mmap not supported");
    }

    void unmap_fragment(size_t first, size_t last) {
        GGML_UNUSED(first);
        GGML_UNUSED(last);
    }
#endif
};

// llama_mlock

struct llama_mlock::impl {
#ifdef _POSIX_MEMLOCK_RANGE
    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

    bool raw_lock(const void * addr, size_t size) const {
        if (!mlock(addr, size)) {
            return true;
        }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
        "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
        "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MEMLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
        "Try increasing RLIMIT_MEMLOCK ('ulimit -l' as root).\n"
#endif

        char* errmsg = std::strerror(errno);
        bool suggest = (errno == ENOMEM);
#if defined(TARGET_OS_VISION) || defined(TARGET_OS_TV) || defined(_AIX)
        // visionOS/tvOS dont't support RLIMIT_MEMLOCK
        // Skip resource limit checks on visionOS/tvOS
        suggest = false;
#else
        struct rlimit lock_limit;
        if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit)) {
            suggest = false;
        }
        if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size)) {
            suggest = false;
        }
#endif

        LLAMA_LOG_WARN("warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                size, this->size, errmsg, suggest ? MLOCK_SUGGESTION : "");
        return false;
    }

    static void raw_unlock(void * addr, size_t size) {
        if (munlock(addr, size)) {
            LLAMA_LOG_WARN("warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
#elif defined(_WIN32)
    static size_t lock_granularity() {
        SYSTEM_INFO si;
        GetSystemInfo(&si);
        return (size_t) si.dwPageSize;
    }

    bool raw_lock(void * ptr, size_t len) const {
        for (int tries = 1; ; tries++) {
            if (VirtualLock(ptr, len)) {
                return true;
            }
            if (tries == 2) {
                LLAMA_LOG_WARN("warning: failed to VirtualLock %zu-byte buffer (after previously locking %zu bytes): %s\n",
                    len, size, llama_format_win_err(GetLastError()).c_str());
                return false;
            }

            SIZE_T min_ws_size, max_ws_size;
            if (!GetProcessWorkingSetSize(GetCurrentProcess(), &min_ws_size, &max_ws_size)) {
                LLAMA_LOG_WARN("warning: GetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
            size_t increment = len + 1048576;
            min_ws_size += increment;
            max_ws_size += increment;
            if (!SetProcessWorkingSetSize(GetCurrentProcess(), min_ws_size, max_ws_size)) {
                LLAMA_LOG_WARN("warning: SetProcessWorkingSetSize failed: %s\n",
                        llama_format_win_err(GetLastError()).c_str());
                return false;
            }
        }
    }

    static void raw_unlock(void * ptr, size_t len) {
        if (!VirtualUnlock(ptr, len)) {
            LLAMA_LOG_WARN("warning: failed to VirtualUnlock buffer: %s\n",
                    llama_format_win_err(GetLastError()).c_str());
        }
    }
#else
    static size_t lock_granularity() {
        return (size_t) 65536;
    }

    bool raw_lock(const void * addr, size_t len) const {
        LLAMA_LOG_WARN("warning: mlock not supported on this system\n");
        return false;
    }

    static void raw_unlock(const void * addr, size_t len) {}
#endif

    impl() : addr(NULL), size(0), failed_already(false) {}

    void init(void * ptr) {
        GGML_ASSERT(addr == NULL && size == 0);
        addr = ptr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > size) {
            if (raw_lock((uint8_t *) addr + size, target_size - size)) {
                size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    void * addr;
    size_t size;

    bool failed_already;
};

llama_mlock::llama_mlock() : pimpl(std::make_unique<impl>()) {}
llama_mlock::~llama_mlock() = default;

void llama_mlock::init(void * ptr) { pimpl->init(ptr); }
void llama_mlock::grow_to(size_t target_size) { pimpl->grow_to(target_size); }

#if defined(_POSIX_MEMLOCK_RANGE) || defined(_WIN32)
const bool llama_mlock::SUPPORTED = true;
#else
const bool llama_mlock::SUPPORTED = false;
#endif

size_t llama_path_max() {
    return PATH_MAX;
}
