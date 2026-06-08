if __name__ == "__main__":
    import importlib.util
    spec = importlib.util.spec_from_file_location("rt","/home/stefan0/bench-nvfp4/scripts/run_throughput.py")
    rt = importlib.util.module_from_spec(spec); spec.loader.exec_module(rt)
    rt.run_worker({"id":"qwen3_6_35b_a3b__nvfp4","family":"Qwen3.6-35B-A3B","type":"moe","format":"nvfp4",
                   "repo":"/home/stefan0/bench-nvfp4/models/qwen3_6_35b_a3b__nvfp4_bf16head","revision":None})
