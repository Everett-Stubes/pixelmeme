"""Does the installed llama-cpp-python actually work on this machine?

Exits 0 if the native library initialises and a grammar is enforced, 1 if
not. The installers run this and fall back to a known-good version, because
"pip install succeeded" says nothing about whether the wheel runs: on this
project's own Windows box, 0.3.36 installed cleanly and then died inside
`llama_backend_init` with an access violation, while 0.3.2 worked.

No model needed -- it never gets as far as loading one.
"""
import sys


def main() -> int:
    try:
        import llama_cpp
    except Exception as exc:
        print("llama-cpp-python not importable: %s" % exc)
        return 1
    print("llama-cpp-python %s" % llama_cpp.__version__)
    try:
        # The call that crashed. It is what Llama() does first.
        llama_cpp.llama_backend_init()
    except Exception as exc:
        print("the native library will not initialise: %s: %s"
              % (type(exc).__name__, exc))
        return 1
    try:
        from llama_cpp import LlamaGrammar
        LlamaGrammar.from_string('root ::= "OK"', verbose=False)
    except Exception as exc:
        print("grammars unavailable: %s: %s" % (type(exc).__name__, exc))
        return 1
    print("ok: backend initialises and grammars compile")
    return 0


if __name__ == "__main__":
    sys.exit(main())
