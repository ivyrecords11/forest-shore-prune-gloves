import csv

def save_params_csv(state_dict, csv_path):
    with open(csv_path, "a", newline="") as file_params:
        writer = csv.writer(file_params)
        writer.writerow(["Layer", "Parameter Name", "Values"])
        for name, param in state_dict.items():
            file_params.write(f"\n{name.split('.')[0]},{name}\n")
            arr = param.detach().cpu().numpy()
            if arr.ndim >= 1:
                for items in arr:
                    if getattr(items, "ndim", 0) >= 1:
                        for it in items:
                            if getattr(it, "ndim", 0) >= 1:
                                for it2 in it:
                                    if getattr(it2, "ndim", 0) >= 1:
                                        for it3 in it2:
                                            file_params.write(f"{it3},")
                                    else:
                                        file_params.write(f"{it2}")
                                    file_params.write("\n")
                            else:
                                file_params.write(f"{it}\n")
                    else:
                        file_params.write(f"{items}\n")
            else:
                file_params.write(f"{arr}\n")