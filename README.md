## What is this program?
This program converts an audio file from 24-bit stereo to 8 bit mono, removing some noise along the way.

## Why?
This is to explore using Tensorflow JS to operate on audio.

## Usage
From the command line:
```
  git clone https://github.com/EvanBurnette/convert-audio-tfjs;
  pnpm install;
  pnpm vite;
```
and command/ctrl + click on the link

## What happened with the two index.js's?
I was making an explanation video and it was getting very long and difficult, so I created an indexVideo.js where I refactored the code to make it far more readable and direct.

I was planning to use the conv1d to find silent sections of the audio file but it had higher rank input and output than I expected. I took this all out for the video because it was making it too long.

## What else did you discover?
1. Websites that are predominantly non-visual are not very interesting to demo and that I wanted a deployment that anyone could play with without setup and installation.

1. ChatGPT can rewrite anything in iambic pentameter:

```
Let start and end in this narrative dwell,
A tale of coding journey I shall tell.
Recording sought, through fetch we make our stand,
From noise-laden waves, it takes its grand.

From start to end, a timer's watchful gaze,
Decoding dance of audio's diverse ways.
With sample rate transformed, a context found,
Sixteen thousand beats, the rhythm's bound.
Decoding span, a moment's fleeting grace,
Measured time records, a fleeting pace.

In fields of info, elements arise,
OG and convert, voices to apprise.
Their inner texts reflect, a story's breath,
Sample rates and channels, life they bequeath.

TensorFlow's realm, a tapestry unfurled,
In TensorFlow's embrace, our journey's twirled.
With tensors of two, stereo's delight,
A rank-2 form, in sound's ethereal flight.
Performance clock, its measured hand deployed,
The tensor's form from memory's vaults enjoyed.

Then rank-1's call, a mono's sweet refrain,
By summing's touch, a solo's pure domain.
From stereo to mono, journey's art,
Through time and values, form's alchemical part.
Absolution sought in values' light,
Negatives transformed to positive's height.

A boolean stage, where silence meets the sound,
Where ones and zeros in tensor's realm are found.
Noise gate's entrance, gated tensors born,
Multiplication's dance, from silence torn.
In union's step, with unity they prance,
Eightfold power's touch, their forms enhance.
A floor beneath their feet, decimals erase,
In TensorFlow's world, they find their space.

From tensor's hold, a final flourish flows,
Array of mono's essence, story's close.
Waveform's symphony, an 8-bit song,
In WaveFile's embrace, the notes prolong.
Info's canvas painted, details known,
Sample rate and channels' essence shown.
By data's transformation, URI's dress,
Convenient for the browser's sweet caress.

Player's stage prepared, the output's role,
A song of bytes and bits, a symphony's soul.
Through "file-saver" art, the download's tale,
"denoised_8bit.wav," its name unveiled.
```

3. ChatGPT is pretty good at generating jokes:

```

1. Why did Evan's audio file go to therapy? Because it had an identity crisis - one moment it wanted to be mono, the next stereo, and in the end, it just couldn't find its true sound!

2. Why did Evan's convolution kernel start a protest? Because it realized that negative ones and positive ones were treated unequally, and it demanded a more balanced audio representation in the tensor world!

3. Why did Evan's audio file apply for a job at the circus? Because it heard there were lots of "sound" opportunities for those who could juggle silence and noise, just like his convoluted TensorFlow attempts!
```
