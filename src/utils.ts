/**
 * Utility functions for audio processing
 */

/**
 * Read audio data from a URL or file path
 * @param url - URL or path to the audio file
 * @param sampling_rate - Target sampling rate for the audio
 * @returns Audio data as Float32Array
 */
export async function read_audio(url: string, sampling_rate: number): Promise<Float32Array> {
    // Fetch the audio file
    const response = await fetch(url);
    const arrayBuffer = await response.arrayBuffer();

    // Decode the audio data
    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    // Resample if necessary
    const channelData = audioBuffer.getChannelData(0);

    if (audioBuffer.sampleRate !== sampling_rate) {
        return resampleAudio(channelData, audioBuffer.sampleRate, sampling_rate);
    }

    return new Float32Array(channelData);
}

/**
 * Resample audio data to a target sampling rate
 * @param audioData - Input audio data
 * @param inputRate - Current sampling rate
 * @param outputRate - Target sampling rate
 * @returns Resampled audio data
 */
function resampleAudio(audioData: Float32Array, inputRate: number, outputRate: number): Float32Array {
    if (inputRate === outputRate) {
        return audioData;
    }

    const ratio = inputRate / outputRate;
    const newLength = Math.round(audioData.length / ratio);
    const result = new Float32Array(newLength);

    for (let i = 0; i < newLength; i++) {
        const position = i * ratio;
        const index = Math.floor(position);
        const fraction = position - index;

        if (index + 1 < audioData.length) {
            result[i] = audioData[index] * (1 - fraction) + audioData[index + 1] * fraction;
        } else {
            result[i] = audioData[index];
        }
    }

    return result;
}

/**
 * Read audio from a Blob (e.g., from MediaRecorder)
 * @param blob - Audio blob
 * @param sampling_rate - Target sampling rate
 * @returns Audio data as Float32Array
 */
export async function read_audio_from_blob(blob: Blob, sampling_rate: number): Promise<Float32Array> {
    const arrayBuffer = await blob.arrayBuffer();

    const audioContext = new (window.AudioContext || (window as any).webkitAudioContext)();
    const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

    const channelData = audioBuffer.getChannelData(0);

    if (audioBuffer.sampleRate !== sampling_rate) {
        return resampleAudio(channelData, audioBuffer.sampleRate, sampling_rate);
    }

    return new Float32Array(channelData);
}
