from torch import load, save

if __name__ == '__main__':
    state_dict = load("models/res-withShift-150-072020.pth")

    new_state_dict = {}

    for key in state_dict.keys():
        new_key = key[7:]
        new_state_dict[new_key] = state_dict[key]

    save(new_state_dict, "models/res-withShift-no_para-150-072020.pth")