import tflite_runtime.interpreter as tflite
import numpy as np
from time import perf_counter

def format_time(seconds):
    if seconds > 1:
        return str(seconds) + " s"
    milliseconds = seconds * 1000
    if milliseconds > 1:
        return str(milliseconds) + " ms"
    microseconds = milliseconds * 1000
    if microseconds > 1:
        return str(microseconds) + " μs"
    return str(microseconds * 1000) + " ns"

dynamic_interpreter = tflite.Interpreter("./tflite_models/dynamic_fashion_mnist.tflite")
full_int_interpreter = tflite.Interpreter("./tflite_models/full_int_fashion_mnist.tflite")
edge_tpu_interpreter = tflite.Interpreter("./tflite_models/full_int_fashion_mnist.tflite", experimental_delegates=[tflite.load_delegate('libedgetpu.so.1')])

input_data = np.reshape(np.load("./fashion_test_data/test_images.npy"), (10000, 1, 28, 28, 1))
output_labels = np.load("./fashion_test_data/test_labels.npy")

dynamic_interpreter.allocate_tensors()
full_int_interpreter.allocate_tensors()
edge_tpu_interpreter.allocate_tensors()

dynamic_input_details = dynamic_interpreter.get_input_details()
full_int_input_details = full_int_interpreter.get_input_details()
edge_tpu_input_details = edge_tpu_interpreter.get_input_details()

dynamic_output_details = dynamic_interpreter.get_output_details()
full_int_output_details = full_int_interpreter.get_output_details()
edge_tpu_output_details = edge_tpu_interpreter.get_output_details()

times = np.empty((3,10000))

dynamic_accuracy = 0
full_int_accuracy = 0
edge_tpu_accuracy = 0

for i in range(10000):
    start = perf_counter()
    dynamic_interpreter.set_tensor(dynamic_input_details[0]['index'], input_data[i].astype("float32"))
    dynamic_interpreter.invoke()
    times[0,i] = perf_counter() - start
    if np.argmax(dynamic_interpreter.get_tensor(dynamic_output_details[0]['index'])) == output_labels[i]:
        dynamic_accuracy += 0.0001
    start = perf_counter()
    full_int_interpreter.set_tensor(full_int_input_details[0]['index'], input_data[i])
    full_int_interpreter.invoke()
    times[1, i] = perf_counter() - start
    if np.argmax(full_int_interpreter.get_tensor(full_int_output_details[0]['index'])) == output_labels[i]:
        full_int_accuracy += 0.0001
    start = perf_counter()
    edge_tpu_interpreter.set_tensor(edge_tpu_input_details[0]['index'], input_data[i])
    edge_tpu_interpreter.invoke()
    times[2, i] = perf_counter() - start
    if np.argmax(edge_tpu_interpreter.get_tensor(edge_tpu_output_details[0]['index'])) == output_labels[i]:
        edge_tpu_accuracy += 0.0001

avgs = np.mean(times, axis=1)
stddevs = np.std(times, axis=1)

print(f"Dynamic Quantization Model: {format_time(np.mean(times[0]))} ± {format_time(np.std(times[0]))} per image (mean ± std. dev. of 10,000 images)")
print(f"Full Integer Quantization Model: {format_time(np.mean(times[1]))} ± {format_time(np.std(times[1]))} per image")
print(f"Edge TPU: {format_time(np.mean(times[2]))} ± {format_time(np.std(times[2]))} per image")
print(f"Dynamic Quantization Model Accuracy: {dynamic_accuracy:.4f}")
print(f"Full Integer Quantization Model Accuracy: {full_int_accuracy:.4f}")
print(f"Edge TPU Model Accuracy: {edge_tpu_accuracy:.4f}")