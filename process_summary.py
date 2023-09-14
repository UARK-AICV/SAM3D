import json

# file_path = "/home/hdhieu/3DSAM-Decoder-1/DATASET_Acdc/output_acdc/sam3d/3d_fullres/Task001_ACDC/sam3d_trainer_acdc__sam3d_Plansv2.1/fold_0/testing_best/summary.json"
file_path = "/home/hdhieu/3DSAM-Decoder-1/DATASET_Tumor/output_tumor/sam3d/3d_fullres/Task003_tumor/sam3d_trainer_tumor__sam3d_Plansv2.1/fold_0/testing_best/summary.json"

if __name__ == "__main__":
    file = open(file_path)
    
    data = json.load(file)
    results = data["results"]
    all_results = results["all"]
    print("#samples: ", len(all_results))
    print(all_results[0].keys())
    print(all_results[0]['0'].keys())

    max_temp = 0
    max_item = 0
    for sample in all_results:
        if sample['1']['Dice'] > max_temp:
            max_temp = sample['1']['Dice']
            max_item = sample
    print(max_item['test'])

    max_temp = 0
    max_item = 0
    for sample in all_results:
        if sample['2']['Dice'] > max_temp:
            max_temp = sample['2']['Dice']
            max_item = sample
    print(max_item['test'])

    max_temp = 0
    max_item = 0
    for sample in all_results:
        if sample['3']['Dice'] > max_temp:
            max_temp = sample['3']['Dice']
            max_item = sample
    print(max_item['test'])

    file.close()