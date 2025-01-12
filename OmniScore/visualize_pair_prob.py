# # Load function
# def load_histogram_data(file_path='histogram_data.npy'):
#     return np.load(file_path, allow_pickle=True).item()

# # Search function to find probability for a given score
# def find_probability(score, data):
#     counts = data['counts']
#     edges = data['edges']
#     for i in range(len(edges) - 1):
#         if edges[i] <= score < edges[i + 1]:
#             return counts[i] / np.sum(counts)  # Probability
#     return 0.0  # If score is out of bounds
# # Example usage
# data = load_histogram_data()
# score = 1.5  # Replace with your score
# probability = find_probability(score, data)
# print(f"Probability for score {score}: {probability:.4f}")

# # first we have to find win video -> real postiion 

# create pair_score.json 
# load pair.json 
# compute a*b then have [prob,prob,xxx,prob]
# duplicate it by max(prob)/prob 



# # plot hist of pair scores and save to "pair_score_hist.png"
# import matplotlib.pyplot as plt
# import numpy as np
# plt.hist(pair_scores, bins=100)
# plt.xlabel('Pair Scores')
# plt.ylabel('Frequency')
# plt.title('Histogram of Pair Scores')
# plt.savefig('pair_score_hist.png')
# plt.show()


# print(np.mean(pair_scores),np.max(pair_scores),np.min(pair_scores))

# def compute_duplicate_factor(scores):
#     max_score = np.max(scores)
#     return (max_score/scores)#.astype(int)

# duplicate_factors = compute_duplicate_factor(pair_scores)
# # duplicata_factors = [int(i) for i in duplicate_factors]
# # empty plot fiture 
# plt.clf()
# plt.hist(duplicate_factors, bins=100)
# plt.xlabel('Duplicate Factors')
# plt.ylabel('Frequency')
# plt.title('Histogram of Duplicate Factors')
# plt.savefig('duplicate_factors_hist.png')

# # print a line plot
# # x axis 1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,1.9
# # y axis bounds 

# bounds = []
# counts = []

# for i in range(10):
#     bound = 1 + i * 0.1
#     count = np.sum(duplicate_factors > bound)
#     print(f"bound:{bound}, count:{count}")
#     bounds.append(bound)
#     counts.append(count)

# # 绘制线性图
# plt.clf()
# plt.plot(bounds, counts, marker='o')
# plt.title('Count of duplicate factors greater than bounds')
# plt.xlabel('Bounds')
# plt.ylabel('Count')
# plt.savefig('bounds.png')
# plt.xticks(np.arange(1.0, 2.0, 0.1))
# plt.grid()
# plt.show()

# to construct a new pair.json 
# we duplicate the item in pair.json when the duplicate factor > 1.4 
# and save it to new_pair.json
# new_pair_data = []
# for i,pair_item in enumerate(pair_data):
#     duplicate_factor = duplicate_factors[i]
#     new_pair_data.append(pair_item)
#     if duplicate_factor > 1.4:
#         new_pair_data.append(pair_item)

# print(len(pair_data),len(new_pair_data))
# new_pair_json_path = "/home/liurt/liurt_data/haoyu/dataset/text2video2-10k/total_score_0P_vbscore/new_pair.json"
# save_to_json(new_pair_data,new_pair_json_path)
