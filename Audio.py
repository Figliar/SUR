import codecs
import torch
from numpy.random import randint
import librosa
import soundfile as sf
from ikrlib import wav16khz2mfcc, train_gmm, logpdf_gmm
import numpy as np
import os.path
import json
import shutil


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


def delete_silence(file):
    audio_file = r'' + file
    audio, sr = librosa.load(audio_file, sr=16000, mono=True)
    clips = librosa.effects.split(audio, top_db=10)
    wav_data = []
    for c in clips:
        data = audio[c[0]: c[1]]
        wav_data.extend(data)
    sf.write(file.replace(str(file.split("/")[-1]), "") + 'no_silence/' + str(file.split("/")[-1]), wav_data, sr)


def adjust_audios_all(dir):
    print("Deleting silence in recordings and saving to temporary audio files...")
    if not os.path.isdir(dir + "/no_silence"):
        os.mkdir(dir + "/no_silence")
    for f in os.listdir(dir):
        if f[-3:] == "wav":
            delete_silence(dir + "/" + f)


def save_gmm_model(weights, mean_values, cov_matrixes, filename):
    dictionary = {
        "weights": weights.tolist(),
        "mean_values": mean_values.tolist(),
        "conv_matrixes": cov_matrixes.tolist()
    }
    json.dump(dictionary, codecs.open(filename, 'w', encoding='utf-8'),
              separators=(',', ':'),
              sort_keys=True,
              indent=4)
    return 0


def load_gmm_model(file):
    obj_text = codecs.open(file, 'r', encoding='utf-8').read()
    data = json.loads(obj_text)
    w = np.array(data["weights"])
    m = np.array(data["mean_values"])
    v = np.array(data["conv_matrixes"])
    return w, m, v


if __name__ == '__main__':
    adjust_audios_all("target_train")
    adjust_audios_all("non_target_train")
    adjust_audios_all("target_dev")
    adjust_audios_all("non_target_dev")

    target_train = wav16khz2mfcc('target_train/no_silence').values()
    non_target_train = wav16khz2mfcc('non_target_train/no_silence').values()
    test_t = wav16khz2mfcc('target_dev/no_silence').values()
    test_n = wav16khz2mfcc('non_target_dev/no_silence').values()

    #######################################
    #         Training GMM_MODEL          #
    #######################################
    target_train = np.vstack(list(target_train))
    non_target_train = np.vstack(list(non_target_train))

    gauss_number_t = 30
    t_means = target_train[randint(1, len(target_train), gauss_number_t)]
    t_cov_matrixes = [np.cov(target_train.T)] * gauss_number_t
    t_weights = np.ones(gauss_number_t) / gauss_number_t

    gauss_number_n = 60
    n_means = non_target_train[randint(1, len(non_target_train), gauss_number_n)]
    n_cov_matrixes = [np.cov(non_target_train.T)] * gauss_number_n
    n_weights = np.ones(gauss_number_n) / gauss_number_n

    P_t = 0.5
    P_n = 0.5

    for i in range(40):
        t_weights, t_means, t_cov_matrixes, TTL_t = train_gmm(target_train, t_weights, t_means, t_cov_matrixes)
        n_weights, n_means, n_cov_matrixes, TTL_n = train_gmm(non_target_train, n_weights, n_means, n_cov_matrixes)
        if i == 10:
            save_gmm_model(t_weights, t_means, t_cov_matrixes, "GMM_target_10.json")
            save_gmm_model(n_weights, n_means, n_cov_matrixes, "GMM_non_target_10.json")
        print("Iteration: %d Total log-likelihood: %f for target. %f for non-target" % (i, TTL_t, TTL_n))

    save_gmm_model(t_weights, t_means, t_cov_matrixes, "GMM_target_40.json")
    save_gmm_model(n_weights, n_means, n_cov_matrixes, "GMM_non_target_40.json")

    score = list()
    for i in test_t:
        i_t = logpdf_gmm(i, t_weights, t_means, t_cov_matrixes)
        i_n = logpdf_gmm(i, n_weights, n_means, n_cov_matrixes)
        score.append((sum(i_t) + np.log(P_t)) - (sum(i_n) + np.log(P_n)))
    print("Accuracy on target: %f" % (np.mean(np.array(score) > 100)))

    score.clear()
    for i in test_n:
        i_t = logpdf_gmm(i, t_weights, t_means, t_cov_matrixes)
        i_n = logpdf_gmm(i, n_weights, n_means, n_cov_matrixes)
        score.append((sum(i_t) + np.log(P_t)) - (sum(i_n) + np.log(P_n)))
    print("Accuracy on non-target: %f" % (np.mean(np.array(score) < 100)))

    # CLEAN UP
    shutil.rmtree("target_train/no_silence")
    shutil.rmtree("target_dev/no_silence")
    shutil.rmtree("non_target_train/no_silence")
    shutil.rmtree("non_target_dev/no_silence")
    exit(0)
