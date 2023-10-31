# ПРОГРАММНЫЙ КОД НЕЙРОННОЙ СЕТИ
import sys
import numpy as np

class Neuron(object):

	def __init__(self, learning_rate=0.1):
		self.weights_0_1 = np.random.normal(0.0, 2 ** -0.5, (3, 2))
		self.weights_1_2 = np.random.normal(0.0, 1, (1, 3))
		self.sigmoid_mapper = np.vectorize(self.sigmoid)
		self.learning_rate = np.array([learning_rate])

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def predict(self, inputs):
		inputs_1 = np.dot(self.weights_0_1, inputs)
		outputs_1 = self.sigmoid_mapper(inputs_1)

		inputs_2 = np.dot(self.weights_1_2, outputs_1)
		outputs_2 = self.sigmoid_mapper(inputs_2)
		return outputs_2

	def train(self, inputs, expected_predict):
		inputs_1 = np.dot(self.weights_0_1, inputs)
		outputs_1 = self.sigmoid_mapper(inputs_1)

		inputs_2 = np.dot(self.weights_1_2, outputs_1)
		outputs_2 = self.sigmoid_mapper(inputs_2)
		actual_predict = outputs_2[0]

		error_layer_2 = np.array([actual_predict - expected_predict])
		gradient_layer_2 = actual_predict * (1 - actual_predict)
		weights_delta_layer_2 = error_layer_2 * gradient_layer_2
		self.weights_1_2 -= (np.dot(weights_delta_layer_2, outputs_1.reshape(1, len(outputs_1)))) * self.learning_rate

		error_layer_1 = weights_delta_layer_2 * self.weights_1_2
		gradient_layer_1 = outputs_1 * (1 - outputs_1)
		weights_delta_layer_1 = error_layer_1 * gradient_layer_1
		self.weights_0_1 -= np.dot(inputs.reshape(len(inputs), 1), weights_delta_layer_1).T  * self.learning_rate
		return self.weights_0_1, self.weights_1_2

def MSE(y, Y):
	return np.mean((y-Y)**2)

train = [
    ([0, 0], 0),
    ([0, 1], 0),
    ([1, 0], 0),
    ([1, 1], 1),
    ]
epochs = 5000
learning_rate = 0.1

network = Neuron(learning_rate=learning_rate)

for e in range(epochs):
	inputs_ = []
	correct_predictions = []
	for input_stat, correct_predict in train:
		network.train(np.array(input_stat), correct_predict)
		inputs_.append(np.array(input_stat))
		correct_predictions.append(np.array(correct_predict))

	train_loss = MSE(network.predict(np.array(inputs_).T), np.array(correct_predictions))
	sys.stdout.write("\rProgress: {}, Training loss: {}\n".format(str(100 * e/float(epochs))[:4], str(train_loss)[:5]))

for input_stat, correct_predict in train:
	print("For input: {} the prediction is: {}, expected: {}".format(
    	str(input_stat),
    	str(network.predict(np.array(input_stat)) > .5),
   		str(correct_predict == 1)))

for input_stat, correct_predict in train:
	print("For input: {} the prediction is: {}, expected: {}".format(
    	str(input_stat),
    	str(network.predict(np.array(input_stat))),
    	str(correct_predict == 1)))
print ("Weights from input to hidden: \n{},\n weights from hidden to output: \n{}".format(
		str(Neuron().weights_0_1),
		str(Neuron().weights_1_2)))

# ПРОГРАММНЫЙ КОД ГОЛОСОВОГО ПОМОЩНИКА
import speech_recognition as sr
import webbrowser
import pyttsx3
import numpy as np
import sys

speak_engine = pyttsx3.init()

def say(words):
	print(words)
	speak_engine.say(words)
	speak_engine.runAndWait()
	speak_engine.stop()

def command():
	r = sr.Recognizer()
	with sr.Microphone() as source:
		print("Говорите")
		r.pause_threshold = 1
		r.adjust_for_ambient_noise(source, duration=1)
		audio = r.listen(source)
	try:
		text = r.recognize_google(audio, language="ru-RU").lower()
		print("Вы сказали: " + text)
	except sr.UnknownValueError:
		say("Я вас не поняла")
		text = command()
	return text

def inp(text):
	input_1 = 0
	input_2 = 0
	for i in range(len(text)):
		if text[i] == '.':
			input_1 = 1
		elif text[i] == 'A' or text[i] == 'a' or text[i] == 'B' or text[i] == 'b' or text[i] == 'C' or text[
			i] == 'c' or \
				text[i] == 'D' or text[i] == 'd' or text[i] == 'E' or text[i] == 'e' or text[i] == 'F' or text[
			i] == 'f' or \
				text[i] == 'G' or text[i] == 'g' or text[i] == 'H' or text[i] == 'h' or text[i] == 'I' or text[
			i] == 'i' or \
				text[i] == 'J' or text[i] == 'j' or text[i] == 'K' or text[i] == 'k' or text[i] == 'L' or text[
			i] == 'l' or \
				text[i] == 'M' or text[i] == 'm' or text[i] == 'N' or text[i] == 'n' or text[i] == 'O' or text[
			i] == 'o' or \
				text[i] == 'P' or text[i] == 'p' or text[i] == 'Q' or text[i] == 'q' or text[i] == 'R' or text[
			i] == 'r' or \
				text[i] == 'S' or text[i] == 's' or text[i] == 'T' or text[i] == 't' or text[i] == 'U' or text[
			i] == 'u' or \
				text[i] == 'V' or text[i] == 'v' or text[i] == 'W' or text[i] == 'w' or text[i] == 'X' or text[
			i] == 'x' or \
				text[i] == 'Y' or text[i] == 'y' or text[i] == 'Z' or text[i] == 'z':
			input_2 = 1
		else:
			input_1 = 0
			input_2 = 0
	i = np.array = ([input_1, input_2])
	return i

class Neuron(object):

	def __init__(self):
		self.weights_0_1 = np.array([[-0.18525204,  0.80828858],
 									 [ 0.83966965,  1.10458251],
 									 [-0.37584951, -0.14104356]])
		self.weights_1_2 = np.array([ 1.67008006,  0.59031368, -0.43641343])
		self.sigmoid_mapper = np.vectorize(self.sigmoid)

	def sigmoid(self, x):
		return 1 / (1 + np.exp(-x))

	def predict(self, i):
		inputs_1 = np.dot(self.weights_0_1, i)
		outputs_1 = self.sigmoid_mapper(inputs_1)

		inputs_2 = np.dot(self.weights_1_2, outputs_1)
		outputs_2 = self.sigmoid_mapper(inputs_2)
		return outputs_2
def makeSomething(text):
	say("Открываю")
	if Neuron().predict(inp(text)) > 0.808:
		url = ('https://' + text)
	else:
		url = ('https://yandex.ru/search/?lr=10735&text=' + text)
	webbrowser.open(url)
	sys.exit()
while True:
	makeSomething(command())
