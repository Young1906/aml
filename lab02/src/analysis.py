from matplotlib import pyplot as plt
import pickle


if __name__ == "__main__":
    with open("hist.pkl", "rb") as f:
        hist = pickle.load(f);

    print(hist)
