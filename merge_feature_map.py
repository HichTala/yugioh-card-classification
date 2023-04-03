import os
import pickle

if __name__ == '__main__':
    feature_map_dir = './feature_maps/feature_maps_partition/'
    feature_map = {}

    for i, files in enumerate(os.listdir(feature_map_dir)):
        abs_path = feature_map_dir + files

        with open(abs_path, "rb") as f:
            tmp = pickle.load(f)
        f.close()

        feature_map.update(tmp)
        del tmp

        if i % 100 == 0:
            print("{} Pickle files merged already !".format(i))

print('{} Cards saved'.format(len(feature_map)))


print("Saving feature map")
savePath = './feature_map.pkl'
with open(savePath, "wb") as f:
    pickle.dump(feature_map, f)
f.close()
print('Feature map saved!')
