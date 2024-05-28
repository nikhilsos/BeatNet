from BeatNet import BeatNet

estimator = BeatNet(1, mode='online', inference_model='PF', plot=['activations'], thread=False)


Output = estimator.process('/home/nikhil/moji/BeatNet/src/BeatNet/test_data/808kick120bpm.mp3')

print(Output)