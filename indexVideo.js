let start, end;
// get recording
const recording_res = await fetch("./recording_with_noise.wav");
const recording_file = await recording_res.arrayBuffer();

start = performance.now();
// decode at sample rate 16khz (original is at 96khz)
const audioCtx = new AudioContext({ sampleRate: 16000 });
const recording = await audioCtx.decodeAudioData(recording_file);
end = performance.now();
console.log(`decoding audio data took ${end - start}ms`);

// get info elements
const og_info = document.getElementById("og_info");
const convert_info = document.getElementById("convert_info");

// update info
og_info.innerText = `${recording.sampleRate}kHz ${recording.numberOfChannels} channel(s)`;

import * as tf from "@tensorflow/tfjs";
const mono_floor_tensor = tf.tidy(() => {
  start = performance.now();
  // create a rank-2 tensor of shape [2, N] two channel audio of arbitrary length (time)
  const stereo_tensor = tf.tensor([
    recording.getChannelData(0),
    recording.getChannelData(1),
  ]);
  end = performance.now();
  stereo_tensor.print();
  console.log(`loading into rank-2 tensor took ${end - start}ms`);

  start = performance.now();
  // create a rank 1 (mono) tensor by summing along the 0th axis
  const mono_tensor = tf.div(stereo_tensor.sum(0), 2);
  end = performance.now();
  mono_tensor.print();
  console.log(`convert to mono took ${end - start}ms`);

  // change all values to positive
  const abs_tensor = mono_tensor.abs();

  // create boolean tensor that has 1 for where audio is playing and a zero for silence
  const bool_tensor = abs_tensor.greater(0.1);
  // const split_bool = bool_tensor.split(100);
  // split_bool.print();

  // noise gated audio
  start = performance.now();
  // const mono_gated_tensor = tf.mul(mono_tensor, bool_tensor);
  const mono_gated_tensor = tf.mul(mono_tensor, bool_tensor);
  const mono_add1_tensor = tf.add(mono_gated_tensor, 1);
  const mono_8bit_tensor = tf.mul(mono_add1_tensor, 128);
  const mono_floor_tensor = tf.floor(mono_8bit_tensor);
  end = performance.now();
  console.log(`noise gate and convert to int took ${end - start}ms`);
  console.log("in tidy", tf.memory().numBytes / 2 ** 20, "MB");
  console.log(tf.memory().numTensors);
  return mono_floor_tensor;
});
const monoArray = await mono_floor_tensor.array();
console.log("after tidy", tf.memory().numBytes / 2 ** 20, "MB");
console.log(tf.memory().numTensors);

// create wave file
import { WaveFile } from "wavefile";
let output = new WaveFile();
output.fromScratch(
  1,
  16000,
  "8",
  // convert to 8 bit
  monoArray
);

//update info
convert_info.innerText = `${output.fmt.sampleRate} kHz ${
  output.fmt.numChannels
} channel(s) ${(output.fmt.byteRate / output.fmt.sampleRate) * 8}-bit`;

// convert into a convenient format for the browser
const output_dataURI = output.toDataURI();
const output_player = document.getElementById("output");
output_player.setAttribute("src", output_dataURI);

import { saveAs } from "file-saver";
const download_btn = document.getElementById("download");
download_btn.addEventListener("click", (e) => {
  saveAs(output_dataURI, "denoised_8bit.wav");
});
