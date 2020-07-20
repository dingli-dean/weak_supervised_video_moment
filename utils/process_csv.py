import pandas
import os

def fulfill_video(src_csv_file, target_csv_file):
    df = pandas.read_csv(src_csv_file)
    video_name = []
    for i, name in enumerate(df['video']):
        if not pandas.isnull(name):
            video_name.append(name)
        else:
            df['video'][i] = video_name[-1]
    print(df['video'][:10])

    df.to_csv(target_csv_file)


src_csv_file = '/media/datasets/ld_data/tacos/tacos_precomp/original_csv/test.csv'
target_csv_file = '/media/datasets/ld_data/tacos/tacos_precomp/test.csv'
fulfill_video(src_csv_file, target_csv_file)



