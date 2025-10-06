import { pipeline, AutomaticSpeechRecognitionPipeline } from "@huggingface/transformers";
import { read_audio, read_audio_from_blob } from "./utils";

/**
 * Whisper transcription service
 */
export class WhisperTranscriber {
    private transcriber: AutomaticSpeechRecognitionPipeline | null = null;
    private isLoading = false;
    private loadingPromise: Promise<void> | null = null;

    /**
     * Initialize the Whisper model
     * @param modelName - Name of the model to use (default: "onnx-community/whisper-tiny.en")
     */
    async initialize(modelName: string = "onnx-community/whisper-tiny.en"): Promise<void> {
        if (this.transcriber) {
            return; // Already initialized
        }

        if (this.isLoading) {
            // Wait for the existing loading operation to complete
            await this.loadingPromise;
            return;
        }

        this.isLoading = true;
        this.loadingPromise = this._loadModel(modelName);
        await this.loadingPromise;
        this.isLoading = false;
    }

    private async _loadModel(modelName: string): Promise<void> {
        console.log(`Loading Whisper model: ${modelName}`);

        this.transcriber = await pipeline(
            "automatic-speech-recognition",
            modelName,
            { dtype: { encoder_model: "fp32", decoder_model_merged: "q4" } }
        );

        console.log("Whisper model loaded successfully");
    }

    /**
     * Transcribe audio from a URL
     * @param audioUrl - URL to the audio file
     * @returns Transcription text
     */
    async transcribeFromUrl(audioUrl: string): Promise<string> {
        if (!this.transcriber) {
            throw new Error("Transcriber not initialized. Call initialize() first.");
        }

        console.time("Transcription time");

        // Load audio data
        const audio = await read_audio(
            audioUrl,
            this.transcriber.processor.feature_extractor.config.sampling_rate
        );

        // Run transcription
        const output = await this.transcriber(audio);

        console.timeEnd("Transcription time");

        return output.text;
    }

    /**
     * Transcribe audio from a Blob (e.g., from MediaRecorder)
     * @param audioBlob - Audio blob to transcribe
     * @returns Transcription text
     */
    async transcribeFromBlob(audioBlob: Blob): Promise<string> {
        if (!this.transcriber) {
            throw new Error("Transcriber not initialized. Call initialize() first.");
        }

        console.time("Transcription time");

        // Load audio data from blob
        const audio = await read_audio_from_blob(
            audioBlob,
            this.transcriber.processor.feature_extractor.config.sampling_rate
        );

        // Run transcription
        const output = await this.transcriber(audio);

        console.timeEnd("Transcription time");

        return output.text;
    }

    /**
     * Check if the transcriber is ready
     */
    isReady(): boolean {
        return this.transcriber !== null;
    }

    /**
     * Unload the model and free up resources
     */
    async dispose(): Promise<void> {
        if (this.transcriber) {
            // The pipeline doesn't have a dispose method, but we can null it
            this.transcriber = null;
            console.log("Whisper model disposed");
        }
    }
}

// Export a singleton instance
export const whisperTranscriber = new WhisperTranscriber();
