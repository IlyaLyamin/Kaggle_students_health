import heapq


def param_parser_by_num(file_name:str, nums:list):
    model_num = 0
    params = {} # {номер модели: словарь с параметрами}
    with open(file_name, "r", encoding="utf-8") as f:
        READ = False
        for line in f:
            if "parameters" in line and model_num in nums:
                READ = True
            elif READ:
                if line.split()[3] == "batch":
                    param = "batch_size"
                    val = line.split()[5]
                else:
                    param, val = line.split()[3:5]

                if model_num not in params.keys():
                    params[model_num] = dict()
                    params[model_num][param] = val
                elif model_num in params.keys():
                    params[model_num][param] = val

            if "shuffle" in line:
                READ = False
                model_num += 1
    return params


def find_best_models(file_name: str, num:int):
    # num --- кол-во моделей
    best_models = {}    # {номер: val}
    with open(file_name, "r", encoding="utf-8") as f:
        model_num = 0
        for line in f:
            if "bal_acc              max_val:" in line: 
                val = float(line.split()[-2].split(":")[-1])
                best_models[model_num] = val
                model_num += 1
        best_models = heapq.nlargest(num, best_models.items(), key=lambda x: x[-1])
    return best_models
                

