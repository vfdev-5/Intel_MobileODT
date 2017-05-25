# Two stage competition FAQ


We'd like to remind you what will happen in the second stage of this competition:

## Why second stage? Why so much trouble?

The spirit of having a second stage is to prevent hand labeling and leaderboard probing of the test data. In order to achieve this, we ask that you upload your source code, including the correct parameters that you used for generating your submission files. This is for you to prove that you have written automated code to create your final submission(s). These "models" that you submit may be examined by Kaggle and the competition host to determine your eligibility to win the competition and claim prizes.

## If I don't have a chance to win, should I upload my model?

Yes. You never know where your team will place on private leaderboard, so it’s worth uploading.

## Will I still be on the leaderboard if I don't submit in the second stage?

You will not. You will need to make a submission in the correct format in the second stage to remain on the leaderboard.

## Will I still be on the leaderboard if I don't upload a model but do submit in the second stage?

Yes. However, if you do not upload a model and finish in a prize position, your team will be removed from the competition standings entirely!

## How do I upload my model?

To ensure that you did write code to produce your results, you are requested to upload your model. To do this, you can go to "More->Team->Your Model" and upload an archive of your code. It is necessary to zip everything as a single file. Please note that this upload link becomes unavailable after the deadline of stage 1, so you will need to upload it before the end of stage 1.

## What should I upload?

When you upload a model, you pack all the code that you are eventually going to use to generate your submission csv file. If your models generate some output files containing the weights, for example, ‘.caffemodel’ or ‘.tfmodel’ files, you are NOT required to submit those. However, you should submit the code used to generate those files. You can typically select two submissions for final scoring, so don't forget to include the code/instructions for reproducing both! It can be totally different code, or it can be the same code with instructions about the modifications you would make to generate each.

## What happens to my pre-trained model?

You only need to include a README file to indicate where you can download the pre-trained files from. For example, if you used vgg16 from keras, you don’t need to upload the weights file, you only need to indicate where you got it from: https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels.h5

## What if my submission is too big?

Our uploader will handle reasonably large files. If you still think your model will be too large, you can instead upload a checksum of your archive file (such as an md5 or sha hash). Note that if you do win, you will still have to upload it. A common reason for folder being too big might be that you included too many of your non-code files. If you upload a checksum, finish in a prize spot, and are unable to subsequently provide an archive that matches the checksum, you will be removed from the competition.

## What if I want my code to stay private if I don’t win

You may upload an encrypted archive and provide the decryption key in the event you win and wish to claim a prize. Alternatively, you can use a checksum, as described above.

## What happens if I want to change something in my code in the second stage?

We expect you may need to make some “non scientific” alterations, such as changes to path names, in order to create your submissions for the second stage. You are allowed to re-train your model (including the stage one data), but your code should not change. You should not be doing any hyper parameter tuning in the second stage. Parameter tuning is permitted as long as it is fully automated.

## Can I upload multiple times?

You may upload as many model files as you wish. Kaggle only keeps the latest upload. Make sure your last upload is the right one!

## Will the number of participants change in the second stage? Will I get a medal? Will I get points?

Yes, the number of participants will be smaller since some people won't submit in the second stage. For the purposes of medals and points calculations, we will use the number of participants in the first stage to calculate points and medals.