# Ported from ComfyUI-IF_Gemini to support Gemini 3.0 API.  Please NOTE: Open Router Functionality has been removed.
This port is for my own personal use, I am sharing back to the community as a number of people have expressed an interest in having this functionality back in CommfyUI.
I am happy to accept PRs and bug reports and will integrate or resolve as possible

## Features

- **Text Generation**: Create content, answer questions, and generate creative text formats
- **Image Analysis**: Describe, analyze, and extract information from images
- **Image Generation**: Generate images with Gemini 3's advanced image generation capabilities
- **Multi-Modal Input**: Combine text and images in your prompts
- **Gemini 3 Enhanced Features**: 
  - Thinking Level control for adjustable reasoning depth
  - Media Resolution control for optimized image/video processing
  - Advanced structured output capabilities
- **Customizable Parameters**: Control temperature, output tokens, and other generation settings
- **Chat Mode**: Maintain conversation history for interactive sessions
- **Batch Processing**: Generate multiple outputs with a single prompt



1. Verify your API key using the "Verify API Key" button in the node

2. Configure the node:
   - For text generation, set "operation_mode" to "analysis" or "generate_text"
   - For image generation, set "operation_mode" to "generate_images"
   - Connect reference images (optional) for style-based generation

3. Set additional parameters as needed:
   - Prompt: Your text instructions
   - Model version: Select appropriate Gemini model
   - Temperature: Controls randomness (0.0-1.0)
   - **Thinking Level** (Gemini 3 only): Controls how much the model "thinks" before responding
     - `none`: Standard response (default)
     - `low`: Brief reasoning (1024 token budget)
     - `medium`: Moderate reasoning (4096 token budget) 
     - `high`: Extensive reasoning (16384 token budget)
   - **Media Resolution** (Gemini 3 only): Controls image/video processing resolution
     - `low`: 512px max dimension
     - `medium`: 1024px max dimension (default)
     - `high`: 2048px max dimension
     - `ultra_high`: 4096px max dimension
   - Chat Mode: Enable conversation history for interactive sessions
   - Clear History: Clear chat history when enabled (works with Chat Mode)
   - Seed: For reproducible results

## Troubleshooting

- If you encounter API key errors, use the "Verify API Key" button to check its validity
- For image safety errors, try modifying your prompt to avoid content that may trigger safety filters
- Ensure your Gemini API has appropriate quotas for your usage

## License

MIT

