import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import colorsys
from matplotlib.lines import Line2D


hue_q2 = np.array([8.05, 10.14, 11.75, 13.45, 12.55, 9.96, 14.06])
sat_q2 = np.array([128.56, 217.9, 202.64, 219.56, 183.19, 114.91, 198.38])
val_q2 = np.array([137.45, 131.68, 118.46, 181.95, 250.25, 242.23, 147.75])

hue_q1 = np.array([0.00, 8.70, 9.63, 12.27, 10.73, 7.89, 13.00])
sat_q1 = np.array([0.00, 204.57, 180.49, 207.98, 161.23, 90.68, 181.38])
val_q1 = np.array([0.25, 7.25, 90.84, 4.22, 222.61, 214.53, 3.88])

hue_q3 = np.array([9.76, 12.33, 13.53, 14.89, 15.58, 12.43, 16.19])
sat_q3 = np.array([156.69, 226.83, 214.06, 228.65, 192.55, 126.38, 210.94])
val_q3 = np.array([159.20, 149.28, 142.02, 201.62, 254.65, 253.83, 163.81])


hue_norm = hue_q2 / 360.0
sat_norm = sat_q2 / 255.0
val_norm = val_q2 / 255.0


colors = [colorsys.hsv_to_rgb(h, s, v) for h, s, v in zip(hue_norm, sat_norm, val_norm)]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

legend_elements = []
# for i in range(len(hue_q2)):
#     ax.scatter(hue_q2[i], sat_q2[i], val_q2[i], c=[colors[i]], s=100, marker='o')
#     legend_label = f'Cl{i+1}: H={hue_q2[i]:.2f}, S={sat_q2[i]:.1f}, V={val_q2[i]:.1f}'
#     legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label,
#                                   markerfacecolor=colors[i], markersize=10))

for i in range(len(hue_q2)):
    ax.scatter(hue_q2[i], sat_q2[i], val_q2[i], c=[colors[i]], s=100, marker='o')
    legend_label = f'Cl{i+1}'
    legend_elements.append(Line2D([0], [0], marker='o', color='w', label=legend_label,
                                  markerfacecolor=colors[i], markersize=10))

ax.set_xlabel('Hue',fontsize=14)
ax.set_ylabel('Saturation',fontsize=14)
ax.set_zlabel('Value',fontsize=14)
ax.set_title('3D Visualization of Cluster Median (Q2) HSV Values',fontsize=16)

ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.05, 1), title='Clusters', fontsize=12, title_fontsize=13)

ax.tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()
