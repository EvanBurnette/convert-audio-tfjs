import { WaveFile } from "wavefile";

const og_info = document.getElementById("og_info");
const convert_info = document.getElementById("convert_info");

const recording_res = await fetch("./recording_with_noise.wav");
const recording_file = await recording_res.arrayBuffer();

// decode at sample rate that out ML model expects i.e. 16khz
const audioCtx = new AudioContext({ sampleRate: 16000 });
// console.log(recording_file);
const recording = await audioCtx.decodeAudioData(recording_file);
console.log(recording);
og_info.innerText = `${recording.sampleRate}kHz ${recording.numberOfChannels} channel(s)`;

// we will have a rank-2 tensor of shape [2, N] two channel audio of arbitrary length (time)
const recording_tensor_stereo = tf.tensor([
  recording.getChannelData(0),
  recording.getChannelData(1),
]);

console.log(recording_tensor_stereo);
const start = performance.now();
const recording_tensor_mono = recording_tensor_stereo.sum(0);
// console.log(await recording_tensor_mono.array());
const end = performance.now();

console.log("stereo to mono took ", end - start, "ms");

// define an onset as a sound following 0.1 seconds of silence
// a sound is when the magnitude is greater than some small threshold
// so if we absolute value all elements and then create a boolean tensor element wise where element > threshold
// we can convolve it with 0.1 second of -1 for our silence range followed by an impulse of +1 for our onset
// so our 1d convolution kernel/filter will look like [-1,-1,-1,-1,-1 ... -1, +1]
// it will give us the onset of the sound plus some offset

const recording_tensor_abs = recording_tensor_mono.abs();
const threshold = tf.scalar(0.001, "float32");
const recording_tensor_bool = recording_tensor_abs.greater(threshold);
console.log("abs val", await recording_tensor_abs.array());
console.log("first sound at", await recording_tensor_bool.argMin().array());

const momentOfSilence = 0.1; //seconds
const samplesOfSilence = Math.floor(momentOfSilence * audioCtx.sampleRate);
console.log(samplesOfSilence);

const momentOfSilenceArr = new Array(samplesOfSilence).fill(-100);
const ones = new Array(audioCtx.sampleRate * 0.01).fill(1);
const onsetFilter = tf.tensor([...momentOfSilenceArr, ...ones]);
onsetFilter.print();

// moment of truth
// let's convolve

const onsets = tf.conv1d(
  [[[recording_tensor_bool]]],
  [[[onsetFilter]]],
  1,
  "same"
);

onsets.print();

// this didn't work because conv1d expects higher rank inputs

// so we can just fall back to the method of counting

const { values, indices } = tf.topk(recording_tensor_bool, 1000);
indices.print();

// get data back into a new audio buffer
// const monoArray = await recording_tensor_mono.array();

const monoArray = await tf
  .mul(recording_tensor_mono, tf.cast(recording_tensor_bool, "float32"))
  .array();

// we're not even using this buffer
const monoRecording = audioCtx.createBuffer(1, monoArray.length, 16000);

let output = new WaveFile();
output.fromScratch(
  1,
  16000,
  "8",
  monoArray.map((n) => Math.floor((256 * (n + 1)) / 2))
);
convert_info.innerText = `${monoRecording.sampleRate} kHz ${monoRecording.numberOfChannels} channel(s)`;

console.log(monoRecording);

const output_dataURI = output.toDataURI();

const output_player = document.getElementById("output");
output_player.setAttribute("src", output_dataURI);
