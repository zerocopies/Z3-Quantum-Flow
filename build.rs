/// Z.1 build.rs
///
/// Compiles llama.cpp b3534 and links it into the binary.
/// b3534 directory layout:
///   ggml/src/   — C sources (ggml.c, ggml-alloc.c, ggml-backend.c, ggml-quants.c, ggml-aarch64.c)
///   ggml/include/ — ggml headers
///   src/         — C++ sources (llama.cpp, llama-vocab.cpp, llama-grammar.cpp, llama-sampling.cpp,
///                                unicode.cpp, unicode-data.cpp)
///   include/     — llama.h
///   common/      — common.cpp, sampling.cpp, grammar-parser.cpp
///
/// Setup (run once):
///   git clone --depth 1 --branch b3534 https://github.com/ggerganov/llama.cpp vendor/llama.cpp

fn main() {
    let llama_dir = std::path::PathBuf::from("vendor/llama.cpp");
    let ggml_src  = llama_dir.join("ggml/src");
    let ggml_inc  = llama_dir.join("ggml/include");
    let llama_inc = llama_dir.join("include");
    let llama_src = llama_dir.join("src");
    let common    = llama_dir.join("common");

    // ── GGML C sources ────────────────────────────────────────────────────────
    let c_sources = [
        "ggml.c",
        "ggml-alloc.c",
        "ggml-backend.c",
        "ggml-quants.c",
        "ggml-aarch64.c",
    ];

    let mut c_build = cc::Build::new();
    c_build
        .include(&ggml_src)
        .include(&ggml_inc)
        .include(&llama_inc)
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")  // uses AVX2 on X240 Haswell
        .flag_if_supported("-DNDEBUG")
        .flag_if_supported("-D_GNU_SOURCE"); // required for NUMA affinity macros
    for src in &c_sources {
        c_build.file(ggml_src.join(src));
    }
    c_build.compile("llama_c");

    // ── llama.cpp C++ sources ─────────────────────────────────────────────────
    let cpp_sources = [
        "src/llama.cpp",
        "src/llama-vocab.cpp",
        "src/llama-grammar.cpp",
        "src/llama-sampling.cpp",
        "src/unicode.cpp",
        "src/unicode-data.cpp",
        "common/common.cpp",
        "common/sampling.cpp",
        "common/grammar-parser.cpp",
    ];

    let mut cpp_build = cc::Build::new();
    cpp_build
        .cpp(true)
        .include(&ggml_src)
        .include(&ggml_inc)
        .include(&llama_inc)
        .include(&llama_src)
        .include(&common)
        .flag_if_supported("-O3")
        .flag_if_supported("-march=native")
        .flag_if_supported("-std=c++17")
        .flag_if_supported("-DNDEBUG")
        .flag_if_supported("-D_GNU_SOURCE");
    for src in &cpp_sources {
        cpp_build.file(llama_dir.join(src));
    }
    cpp_build.compile("llama_cpp");

    println!("cargo:rustc-link-lib=static=llama_c");
    println!("cargo:rustc-link-lib=static=llama_cpp");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rerun-if-changed=vendor/llama.cpp");
}
