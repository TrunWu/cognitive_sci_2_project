#visualsation of matrix
import numpy as np
import matplotlib.pyplot as plt 

private = np.array([[0.60081466, 0.01629328, 0.10590631, 0.03258656, 0.12219959,
        0.01832994, 0.10386965],
       [0.25454545, 0.54545455, 0.10909091, 0.        , 0.03636364,
        0.01818182, 0.03636364],
       [0.16287879, 0.        , 0.43560606, 0.03219697, 0.18181818,
        0.10416667, 0.08333333],
       [0.03071672, 0.        , 0.01934016, 0.86916951, 0.02502844,
        0.01706485, 0.03868032],
       [0.1026936 , 0.00673401, 0.11784512, 0.04713805, 0.54040404,
        0.00841751, 0.17676768],
       [0.03365385, 0.        , 0.12019231, 0.04326923, 0.02884615,
        0.75      , 0.02403846],
       [0.05591054, 0.00319489, 0.08306709, 0.05111821, 0.15335463,
        0.00958466, 0.64376997]])

public = np.array([[0.6124197 , 0.01284797, 0.08993576, 0.04282655, 0.11777302,
        0.02997859, 0.09421842],
       [0.33928571, 0.5       , 0.07142857, 0.03571429, 0.03571429,
        0.        , 0.01785714],
       [0.14919355, 0.01008065, 0.38306452, 0.03225806, 0.2358871 ,
        0.06854839, 0.12096774],
       [0.02458101, 0.00111732, 0.02122905, 0.83910615, 0.01899441,
        0.02346369, 0.07150838],
       [0.13169985, 0.01071975, 0.09341501, 0.05359877, 0.53598775,
        0.01990812, 0.15467075],
       [0.03855422, 0.        , 0.07228916, 0.04578313, 0.02409639,
        0.79036145, 0.02891566],
       [0.08237232, 0.00329489, 0.05271829, 0.092257  , 0.13673806,
        0.01976936, 0.61285008]])

emotic = np.array([
	[0.71232877, 0.04109589, 0.02739726, 0.20547945, 0.01369863, 0.        ],
    [0.27272727, 0.22727273, 0.        , 0.31818182, 0.18181818,
    0.        ],
    [0.7       , 0.1       , 0.        , 0.2       , 0.        ,
        0.        ],
    [0.37209302, 0.        , 0.        , 0.60465116, 0.02325581,
        0.        ],
    [0.36363636, 0.        , 0.        , 0.39393939, 0.24242424,
        0.        ],
    [0.26666667, 0.06666667, 0.        , 0.53333333, 0.13333333,
        0.        ]])

ck = np.array([[0.31707317, 0.        , 0.04878049, 0.02439024, 0.17073171,
        0.02439024, 0.41463415],
       [0.32727273, 0.07272727, 0.        , 0.09090909, 0.07272727,
        0.        , 0.43636364],
       [0.17391304, 0.        , 0.04347826, 0.13043478, 0.17391304,
        0.08695652, 0.39130435],
       [0.        , 0.        , 0.01587302, 0.74603175, 0.03174603,
        0.        , 0.20634921],
       [0.        , 0.        , 0.03846154, 0.        , 0.61538462,
        0.        , 0.34615385],
       [0.07894737, 0.        , 0.01315789, 0.02631579, 0.02631579,
        0.47368421, 0.38157895],
       [0.        , 0.        , 0.        , 0.        , 0.        ,
        0.        , 0.        ]])

def visualize(matrix, dataset_title, label_class,x_label_text = 'Predicted Lables',y_label_text = 'Actual Labels'):
  fig,ax = plt.subplots()
	# round the accuracy in matrix after 2 digits after dot
  show_matrix = np.around((matrix), decimals=2)
  im = ax.imshow(show_matrix)
  #set the ticks
  ax.set_xticks(np.arange(len(label_class)))
  ax.set_yticks(np.arange(len(label_class)))
  ax.set_xticklabels(label_class)
  ax.set_yticklabels(label_class)
	#rotate the tick labels and set alignments
  plt.setp(ax.get_xticklabels(), rotation=45, ha='right', rotation_mode='anchor')
	#set title
  ax.set_title(dataset_title)
	#set x label
  ax.set_xlabel(x_label_text)
  ax.set_ylabel(y_label_text)
	#create text annotation
  for x in range(len(label_class)):
    for y in range(len(label_class)):
      text = ax.text(y, x, show_matrix[x, y], ha='center', va='center', color='w')
  #create color bar
  cbar = ax.figure.colorbar(im, ax=ax)
  #cbar = ax.set_ylabel()
  fig.tight_layout()
  plt.show()

#run it 
title_private = 'Normalized Confusion Matrix of Private Test Dataset'
title_public = 'Normalized Confusion Matrix of Public Test Dataset'
title_emotic = 'Normalized Confusion Matrix of Emotic Test Dataset'
title_ck = 'Normalized Confusion Matrix of CK+ Test Dataset'
labels = np.asarray(['angry', 'disgust', 'fear','happy', 'sad', 'surprise', 'neutral'])
emotic_labels = np.asarray(['angry', 'disgust', 'fear','happy', 'sad', 'surprise'])
visualize(private, title_private, labels)
visualize(public, title_public, labels)
visualize(emotic, title_emotic, emotic_labels)
visualize(ck, title_ck, labels)
#print(np.around(private, decimals=2))