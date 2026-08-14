for figure 10 we have the max trainabel model for 1 gpu and 2 gpus. but honestly i wanna change it to a short seq length and a longer sqeunce legnth and then only do one gpu so that we can see the max trainable model parameter size to be more obvious?
Cause the issue is that when we use 2 gpus ... we are NOT sharding the model the max trainab;e model size becomes limited and notn sacling to twice .. so this woud mak the reviewer question like "huh so more gous cannot scale the model? hm" rihgt? let me know?


for the current throughput plots lets 

