import torch

from Audio import load_gmm_model, adjust_audios_all
from train_image_model import load_model
from ikrlib import logpdf_gmm, wav16khz2mfcc, png2fea
import numpy as np
import argparse
import shutil
from torch import Tensor

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluation script for SUR")
    parser.add_argument("-D", "--dir", required=True, default="NO_DIR_PASSED", help="Directory with data to process")
    args = parser.parse_args()
    if args.dir == "NO_DIR_PASSED":
        print("No directory passed. PLease call script with \"-D <directory_name>\"")

    print("Evaluating GMM_MODEL_10...")
    ###################################
    #        Eval GMM_MODEL_10        #
    ###################################
    weights_t, means_t, cov_ms_t = load_gmm_model("GMM_target_10.json")
    weights_n, means_n, cov_ms_n = load_gmm_model("GMM_non_target_10.json")
    P_target = 0.5
    P_non_target = 0.5

    adjust_audios_all(args.dir)
    data_gmm = wav16khz2mfcc(args.dir + "/no_silence")

    score_gmm = list()
    gmm_file = open("gmm_10.txt", "wb")
    for key in data_gmm.keys():
        i_t = logpdf_gmm(data_gmm[key], weights_t, means_t, cov_ms_t)
        i_n = logpdf_gmm(data_gmm[key], weights_n, means_n, cov_ms_n)
        score_gmm.append((sum(i_t) + np.log(P_target)) - (sum(i_n) + np.log(P_non_target)))
        score = (sum(i_t) + np.log(P_target)) - (sum(i_n) + np.log(P_non_target))
        name = key.split("/")[-1].replace(".wav", "")
        score = score - 100
        if score > 0:
            decision = 1
        else:
            decision = 0
        output = name + " " + str(score) + " " + str(decision) + "\n"
        output = output.encode('ascii')
        gmm_file.write(output)
    gmm_file.close()

    print("Evaluating GMM_MODEL_40...")
    ###################################
    #        Eval GMM_MODEL_40        #
    ###################################
    weights_t, means_t, cov_ms_t = load_gmm_model("GMM_target_40.json")
    weights_n, means_n, cov_ms_n = load_gmm_model("GMM_non_target_40.json")

    score_gmm = []
    gmm_file = open("gmm_40.txt", "wb")
    for key in data_gmm.keys():
        i_t = logpdf_gmm(data_gmm[key], weights_t, means_t, cov_ms_t)
        i_n = logpdf_gmm(data_gmm[key], weights_n, means_n, cov_ms_n)
        score_gmm.append((sum(i_t) + np.log(P_target)) - (sum(i_n) + np.log(P_non_target)))
        score = (sum(i_t) + np.log(P_target)) - (sum(i_n) + np.log(P_non_target))
        name = key.split("/")[-1].replace(".wav", "")
        score = score - 100
        if score > 0:
            decision = 1
        else:
            decision = 0
        output = name + " " + str(score) + " " + str(decision) + "\n"
        output = output.encode('ascii')
        gmm_file.write(output)
    gmm_file.close()

    print("Evalluating IMAGE_MODEL...")
    ################################
    #       Eval IMAGE_MODEL       #
    ################################
    image_model = load_model("IMAGE_MODEL")
    score_image = list()
    image_file = open("image_model_fin.txt", "wb")
    data_image_linear = png2fea(args.dir)
    with torch.no_grad():
        for key in data_image_linear.keys():
            image = data_image_linear[key].mean(axis=2).reshape(-1, 80*80)
            name = key.split("/")[-1].replace(".png", "")
            score = image_model(Tensor(image)).item()
            score_image.append(score)
            score = score - 0.225
            if score <= 0:
                decision = 0
            else:
                decision = 1
            output = name + " " + str(score) + " " + str(decision) + "\n"
            output = output.encode('ascii')
            image_file.write(output)
    image_file.close()

    print("Evaluating IMAGE_MODEL_PLUS..")
    #####################################
    #       Eval IMAGE_MODEL_PLUS       #
    #####################################
    image_model_plus = load_model("IMAGE_MODEL_PLUS")
    score_image.clear()
    image_file = open("image_model_plus.txt", "wb")
    data_image_linear = png2fea(args.dir)
    with torch.no_grad():
        for key in data_image_linear.keys():
            image = data_image_linear[key].mean(axis=2).reshape(-1, 80 * 80)
            name = key.split("/")[-1].replace(".png", "")
            score = image_model_plus(Tensor(image)).item()
            score_image.append(score)
            score = score - 0.225
            if score <= 0:
                decision = 0
            else:
                decision = 1
            output = name + " " + str(score) + " " + str(decision) + "\n"
            output = output.encode('ascii')
            image_file.write(output)
    image_file.close()

    print("Evaluating combination of IMAGE_MODEL and GMM..")
    ########################################################
    #       Eval combination of GMM and IMAGE models       #
    #                (Purely experimental)                 #
    ########################################################
    combined_file = open("combined.txt", "wb")
    combined_score = list()
    score_gmm = []

    for key in data_image_linear.keys():
        key2 = key.replace(".png", ".wav").replace(args.dir, args.dir + "/no_silence")
        image = data_image_linear[key].mean(axis=2).reshape(-1, 80*80)
        name = key.split("/")[-1].replace(".png", "")
        image_score = image_model(Tensor(image)).item()
        score_image.append(image_score)
        i_t = logpdf_gmm(data_gmm[key2], weights_t, means_t, cov_ms_t)
        i_n = logpdf_gmm(data_gmm[key2], weights_n, means_n, cov_ms_n)
        gmm_score = (sum(i_t) + np.log(P_target)) - (sum(i_n) + np.log(P_non_target))
        score_gmm.append(gmm_score)

    score_gmm = (score_gmm - np.min(score_gmm)) / (np.max(score_gmm) - np.min(score_gmm))
    score_image = (score_image - np.min(score_image)) / (np.max(score_image) - np.min(score_image))

    for i, key in enumerate(data_image_linear.keys()):
        name = key.split("/")[-1].replace(".png", "")
        s_g = score_gmm[i]
        s_i = score_image[i]
        score = s_g * s_i
        if score > 0.364:
            decision = 1
        else:
            decision = 0
        output = name + " " + str(score) + " " + str(decision) + "\n"
        output = output.encode('ascii')
        combined_file.write(output)
    combined_file.close()

    shutil.rmtree(args.dir + "/no_silence")

    exit(0)
