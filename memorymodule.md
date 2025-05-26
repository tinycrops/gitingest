# Repository Analysis

## Summary

```
Repository: tinycrops/starter-applets
Files analyzed: 7

Estimated tokens: 22.2k
```

## Important Files

```
Directory structure:
└── tinycrops-starter-applets/
    ├── docs/
    └── video-watcher/
        ├── memory/
        ├── server/
        │   ├── index.mjs
        │   ├── memory-manager.mjs
        │   └── video-processor.mjs
        ├── src/
        │   ├── App.jsx
        │   ├── api.js
        │   ├── main.jsx
        │   ├── components/
        │   │   └── VideoDiscussion.jsx
        │   └── styles/
        └── .vite/
            ├── deps_temp_322a5846/
            ├── deps_temp_4b531892/
            └── deps_temp_990e7c0d/

```

## Content

```


================================================
File: video-watcher/server/index.mjs
================================================
import express from 'express';
import ViteExpress from 'vite-express';
import fs from 'fs/promises';
import { createReadStream } from 'fs';
import path from 'path';
import chokidar from 'chokidar';
import { fileURLToPath } from 'url';
import { analyzeVideo, saveToDataset, genAI, generateThumbnail } from './video-processor.mjs';
import memoryManager from './memory-manager.mjs';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// Configuration
const WATCH_FOLDER = process.env.VIDEO_WATCH_FOLDER || 'Q:\\';
const DATASET_FOLDER = process.env.VIDEO_DATASET_FOLDER || path.join(process.env.HOME || process.env.USERPROFILE, 'video-dataset');

const THUMBNAIL_FOLDER = path.join(DATASET_FOLDER, 'thumbnails');

// Keep track of processed videos to avoid reprocessing
const processedVideos = new Set();

// Track files that are being processed to avoid duplicate processing
const processingFiles = new Set();

// Function to check if file size has stabilized (recording has stopped)
async function isFileStable(filePath) {
  try {
    // Get initial file size
    const initialStats = await fs.stat(filePath);
    const initialSize = initialStats.size;
    
    // Wait 3 seconds
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    // Get new file size
    const newStats = await fs.stat(filePath);
    const newSize = newStats.size;
    
    // If sizes match, file is no longer being written to
    const isStable = initialSize === newSize;
    console.log(`File ${path.basename(filePath)} size check: ${initialSize} -> ${newSize}, stable: ${isStable}`);
    return isStable;
  } catch (error) {
    console.error(`Error checking file stability: ${error.message}`);
    return false;
  }
}

// Load already processed videos
async function loadProcessedVideos() {
  try {
    // Check if dataset folder exists
    try {
      await fs.access(DATASET_FOLDER);
    } catch (error) {
      return; // If folder doesn't exist yet, there are no processed videos
    }
    
    // Get all json files from dataset folder
    const files = await fs.readdir(DATASET_FOLDER);
    
    for (const file of files) {
      if (file.endsWith('.json')) {
        try {
          const data = await fs.readFile(path.join(DATASET_FOLDER, file), 'utf-8');
          const videoData = JSON.parse(data);
          
          // Add the video filename to the processed set
          if (videoData && videoData.videoFileName) {
            processedVideos.add(videoData.videoFileName);
            console.log(`Marked as already processed: ${videoData.videoFileName}`);
          }
        } catch (error) {
          console.error(`Error reading processed video data: ${file}`, error);
        }
      }
    }
    
    console.log(`Loaded ${processedVideos.size} already processed videos`);
  } catch (error) {
    console.error('Error loading processed videos:', error);
  }
}

// Ensure dataset directory exists
async function ensureDirectoryExists(directory) {
  try {
    await fs.mkdir(directory, { recursive: true });
    console.log(`Created directory: ${directory}`);
  } catch (error) {
    console.error(`Error creating directory: ${directory}`, error);
  }
}

// Ensure thumbnail folder exists
async function ensureThumbnailFolder() {
  await ensureDirectoryExists(THUMBNAIL_FOLDER);
}

// Check all videos for missing thumbnails and generate them if needed
async function generateMissingThumbnails() {
  await ensureThumbnailFolder();
  const files = await fs.readdir(DATASET_FOLDER);
  const videoFiles = files.filter(f => f.endsWith('.json'));
  for (const jsonFile of videoFiles) {
    try {
      const data = JSON.parse(await fs.readFile(path.join(DATASET_FOLDER, jsonFile), 'utf-8'));
      const videoFileName = data.videoFileName;
      if (!videoFileName) continue;
      const videoPath = path.join(WATCH_FOLDER, videoFileName);
      const thumbName = path.parse(videoFileName).name + '.jpg';
      const thumbPath = path.join(THUMBNAIL_FOLDER, thumbName);
      try {
        await fs.access(thumbPath);
        // Thumbnail exists
      } catch {
        // Thumbnail missing, try to generate
        try {
          await generateThumbnail(videoPath, thumbPath);
          console.log(`Generated missing thumbnail for ${videoFileName}`);
        } catch (err) {
          console.warn(`Could not generate thumbnail for ${videoFileName}:`, err.message);
        }
      }
    } catch (err) {
      console.warn(`Could not process dataset entry ${jsonFile}:`, err.message);
    }
  }
}

// Set up Express
const app = express();
app.use(express.json());

// API endpoints
app.get('/api/status', (req, res) => {
  res.json({
    status: 'active',
    watchFolder: WATCH_FOLDER,
    datasetFolder: DATASET_FOLDER,
    processedCount: processedVideos.size
  });
});

// Add memory API endpoint
app.get('/api/memory', (req, res) => {
  try {
    const memoryState = memoryManager.getMemoryState();
    res.json({
      shortTermMemory: memoryState.shortTermMemory,
      longTermMemory: memoryState.longTermMemory,
      workingMemory: memoryState.workingMemory
    });
  } catch (error) {
    console.error('Error fetching memory state:', error);
    res.status(500).json({ error: 'Failed to retrieve memory state' });
  }
});

// Endpoint for conversational memory interface
app.post('/api/memory/query', async (req, res) => {
  try {
    if (!req.body.query) {
      return res.status(400).json({ error: 'Query parameter is required' });
    }
    
    const response = await memoryManager.conversationalMemoryQuery(req.body.query);
    res.json({ response });
  } catch (error) {
    console.error('Error processing memory query:', error);
    res.status(500).json({ error: 'Failed to process memory query' });
  }
});

// New endpoint to manually process a video
app.get('/api/process/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    const filePath = path.join(WATCH_FOLDER, filename);
    
    console.log(`Manually processing video: ${filePath}`);
    
    // Check if file exists
    try {
      await fs.access(filePath);
    } catch (error) {
      return res.status(404).json({ error: `File not found: ${filePath}` });
    }
    
    // Process the video
    const result = await analyzeVideo(filePath);
    const saveResult = await saveToDataset(filePath, result, DATASET_FOLDER);
    console.log(`Manually processed and saved analysis for: ${filePath}`);
    
    // Process analysis results with memory manager
    await memoryManager.processNewAnalysis(result);
    
    // Mark as processed
    processedVideos.add(filename);
    
    // Generate thumbnail
    await ensureThumbnailFolder();
    const thumbName = path.parse(filename).name + '.jpg';
    const thumbPath = path.join(THUMBNAIL_FOLDER, thumbName);
    await generateThumbnail(filePath, thumbPath);
    
    res.json({ 
      success: true, 
      videoPath: filePath,
      datasetPath: saveResult.datasetPath 
    });
  } catch (error) {
    console.error('Error processing video:', error);
    res.status(500).json({ error: error.message });
  }
});

app.get('/api/videos', async (req, res) => {
  try {
    const files = await fs.readdir(DATASET_FOLDER);
    const videos = [];
    
    for (const file of files) {
      if (file.endsWith('.json')) {
        const data = await fs.readFile(path.join(DATASET_FOLDER, file), 'utf-8');
        videos.push(JSON.parse(data));
      }
    }
    
    res.json({ videos });
  } catch (error) {
    console.error('Error reading videos:', error);
    res.status(500).json({ error: 'Failed to read videos' });
  }
});

// Function to find unprocessed videos
async function scanForMissedVideos() {
  try {
    console.log('Scanning for missed videos...');
    
    // Get all mp4 files in the watch folder
    const files = await fs.readdir(WATCH_FOLDER);
    const videoFiles = files.filter(file => file.endsWith('.mp4'));
    
    let missedCount = 0;
    let reprocessedCount = 0;
    
    // First, check for failed processing attempts in the dataset
    try {
      const datasetFiles = await fs.readdir(DATASET_FOLDER);
      const jsonFiles = datasetFiles.filter(file => file.endsWith('.json'));
      
      for (const jsonFile of jsonFiles) {
        try {
          const content = await fs.readFile(path.join(DATASET_FOLDER, jsonFile), 'utf-8');
          const data = JSON.parse(content);
          
          // Check if this is a failed processing attempt
          if (data.analysis?.error === 'File never reached ACTIVE state after multiple attempts') {
            const videoPath = data.videoPath;
            const videoFileName = data.videoFileName;
            
            // Skip if this file is currently being processed
            if (processingFiles.has(videoPath)) {
              console.log(`File ${videoFileName} is already being processed, skipping.`);
              continue;
            }
            
            processingFiles.add(videoPath);
            
            try {
              // Check if file is stable before processing
              console.log(`Checking if failed video is stable: ${videoFileName}`);
              let stable = await isFileStable(videoPath);
              
              if (stable) {
                console.log(`Reprocessing previously failed video: ${videoPath}`);
                const result = await analyzeVideo(videoPath);
                await saveToDataset(videoPath, result, DATASET_FOLDER);
                console.log(`Successfully reprocessed and updated analysis for: ${videoPath}`);
                
                // Process analysis results with memory manager
                await memoryManager.processNewAnalysis(result);
                
                // Mark as processed
                processedVideos.add(videoFileName);
                reprocessedCount++;
                
                // Generate thumbnail
                await ensureThumbnailFolder();
                const thumbName = path.parse(videoFileName).name + '.jpg';
                const thumbPath = path.join(THUMBNAIL_FOLDER, thumbName);
                await generateThumbnail(videoPath, thumbPath);
              } else {
                console.log(`Failed video ${videoFileName} is still being written, will try again later.`);
              }
            } catch (error) {
              console.error(`Error reprocessing failed video ${videoPath}:`, error);
            } finally {
              processingFiles.delete(videoPath);
            }
          }
        } catch (error) {
          console.error(`Error reading/parsing dataset file ${jsonFile}:`, error);
        }
      }
    } catch (error) {
      console.error('Error scanning dataset for failed videos:', error);
    }
    
    // Then check for completely unprocessed videos
    for (const videoFile of videoFiles) {
      if (!processedVideos.has(videoFile)) {
        missedCount++;
        console.log(`Found missed video: ${videoFile}`);
        
        // Process the video if it's stable (not currently being recorded)
        const filePath = path.join(WATCH_FOLDER, videoFile);
        
        // Skip if this file is currently being processed
        if (processingFiles.has(filePath)) {
          console.log(`File ${videoFile} is already being processed, skipping.`);
          continue;
        }
        
        processingFiles.add(filePath);
        
        try {
          // Check if file is stable before processing
          console.log(`Checking if missed video is stable: ${videoFile}`);
          let stable = await isFileStable(filePath);
          
          if (stable) {
            console.log(`Processing missed video: ${filePath}`);
            const result = await analyzeVideo(filePath);
            await saveToDataset(filePath, result, DATASET_FOLDER);
            console.log(`Processed and saved analysis for missed video: ${filePath}`);
            
            // Process analysis results with memory manager
            await memoryManager.processNewAnalysis(result);
            
            // Mark as processed
            processedVideos.add(videoFile);
            
            // Generate thumbnail
            await ensureThumbnailFolder();
            const thumbName = path.parse(videoFile).name + '.jpg';
            const thumbPath = path.join(THUMBNAIL_FOLDER, thumbName);
            await generateThumbnail(filePath, thumbPath);
          } else {
            console.log(`Missed video ${videoFile} is still being written, will try again later.`);
          }
        } catch (error) {
          console.error(`Error processing missed video ${filePath}:`, error);
        } finally {
          processingFiles.delete(filePath);
        }
      }
    }
    
    console.log(`Scan complete. Found ${missedCount} missed videos and reprocessed ${reprocessedCount} failed videos.`);
  } catch (error) {
    console.error('Error scanning for missed videos:', error);
  }
}

// Set up file watcher
async function setupWatcher() {
  // Initialize memory manager
  try {
    await memoryManager.initialize();
    console.log('Memory manager initialized');
  } catch (error) {
    console.error('Error initializing memory manager:', error);
  }
  
  await ensureDirectoryExists(DATASET_FOLDER);
  await loadProcessedVideos();
  await ensureThumbnailFolder();
  // Run thumbnail generation in the background
  console.log('Starting background thumbnail generation for missing thumbnails...');
  generateMissingThumbnails().then(() => {
    console.log('Background thumbnail generation complete.');
  }).catch(err => {
    console.warn('Background thumbnail generation failed:', err.message);
  });
  
  // Scan for missed videos on startup
  await scanForMissedVideos();
  
  const watcher = chokidar.watch(WATCH_FOLDER, {
    ignored: /(^|[\/\\])\../, // Ignore hidden files
    persistent: true,
    awaitWriteFinish: {
      stabilityThreshold: 5000,
      pollInterval: 1000
    }
  });
  
  watcher
    .on('add', async (filePath) => {
      const fileName = path.basename(filePath);
      
      // Skip if this file has already been processed or is currently being processed
      if (!filePath.endsWith('.mp4') || processedVideos.has(fileName) || processingFiles.has(filePath)) {
        return;
      }
      
      console.log(`New video detected: ${filePath}`);
      processingFiles.add(filePath);
      
      try {
        // First wait for chokidar's stabilityThreshold
        await new Promise(resolve => setTimeout(resolve, 6000));
        
        // Then do our own check to make sure recording has stopped
        console.log(`Checking if recording has completed for: ${filePath}`);
        let stable = false;
        let attempts = 0;
        
        // Keep checking until file size stabilizes (up to 5 attempts)
        while (!stable && attempts < 5) {
          stable = await isFileStable(filePath);
          if (!stable) {
            console.log(`File ${path.basename(filePath)} is still being written, waiting...`);
            await new Promise(resolve => setTimeout(resolve, 3000));
            attempts++;
          }
        }
        
        if (stable) {
          console.log(`Recording complete, processing: ${filePath}`);
          const result = await analyzeVideo(filePath);
          await saveToDataset(filePath, result, DATASET_FOLDER);
          console.log(`Processed and saved analysis for: ${filePath}`);
          
          // Process analysis results with memory manager
          await memoryManager.processNewAnalysis(result);
          
          // Mark as processed
          processedVideos.add(fileName);
          
          // Generate thumbnail
          await ensureThumbnailFolder();
          const thumbName = path.parse(fileName).name + '.jpg';
          const thumbPath = path.join(THUMBNAIL_FOLDER, thumbName);
          await generateThumbnail(filePath, thumbPath);
        } else {
          console.log(`File ${path.basename(filePath)} never stabilized, skipping processing`);
        }
      } catch (error) {
        console.error(`Error processing video ${filePath}:`, error);
      } finally {
        processingFiles.delete(filePath);
      }
    });
    
  // Setup periodic scan for missed videos (every 2 minutes)
  const scanInterval = setInterval(() => {
    scanForMissedVideos();
  }, 2 * 60 * 1000);
  
  // Cleanup interval on process exit
  process.on('SIGINT', () => {
    clearInterval(scanInterval);
    process.exit(0);
  });
  
  console.log('File watcher initialized.');
}

// Function to search video analyses based on a natural language query
async function searchVideoAnalyses(query) {
  console.log(`Received search query: ${query}`);
  try {
    // Read all analysis files from the dataset folder
    const files = await fs.readdir(DATASET_FOLDER);
    const jsonFiles = files.filter(file => file.endsWith('.json'));
    
    if (jsonFiles.length === 0) {
      return [];
    }
    
    // Read and parse each JSON file
    const analysesData = [];
    for (const file of jsonFiles) {
      try {
        const content = await fs.readFile(path.join(DATASET_FOLDER, file), 'utf-8');
        const data = JSON.parse(content);
        analysesData.push({ filename: file, data });
      } catch (error) {
        console.error(`Error parsing JSON file ${file}:`, error);
      }
    }
    
    // Prepare data for relevance check
    const videoInfos = [];
    for (const item of analysesData) {
      const { data } = item;
      
      // Extract key text content from the analysis
      let textContent = '';
      
      // Include summary if available
      if (data.analysis?.summary) {
        textContent += `Summary: ${data.analysis.summary}\n\n`;
      }
      
      // Include transcript if available
      if (data.analysis?.transcript) {
        textContent += `Transcript: ${data.analysis.transcript}\n\n`;
      }
      
      // Include topics if available
      if (data.analysis?.topics && data.analysis.topics.length > 0) {
        textContent += `Topics: ${data.analysis.topics.join(', ')}\n\n`;
      }
      
      // Include inferred insights if available
      if (data.analysis?.inferred_insights && data.analysis.inferred_insights.length > 0) {
        textContent += 'Insights:\n';
        data.analysis.inferred_insights.forEach(insight => {
          textContent += `- ${insight.insight} (Basis: ${insight.basis})\n`;
        });
        textContent += '\n';
      }
      
      videoInfos.push({
        filename: item.filename,
        videoPath: data.videoPath,
        videoFileName: data.videoFileName,
        processedAt: data.processedAt,
        textContent
      });
    }
    
    // Use Gemini to evaluate relevance in batches
    const model = genAI.getGenerativeModel({ model: 'gemini-1.5-flash' });
    const relevantVideos = [];
    
    // Process videos in batches of 10
    const BATCH_SIZE = 10;
    for (let i = 0; i < videoInfos.length; i += BATCH_SIZE) {
      const batch = videoInfos.slice(i, i + BATCH_SIZE);
      
      // Create a combined prompt for the batch
      const batchPrompt = `
        Analyze the following video analyses and determine their relevance to the user's question: "${query}"

        For each video analysis below, determine if it is relevant and provide a relevance score and justification.
        Respond with a JSON array where each element contains:
        {
          "filename": "The filename of the video",
          "is_relevant": boolean,
          "relevance_score": number (0.0 to 1.0),
          "justification": "Brief explanation (1-2 sentences)"
        }

        Video Analyses:
        ${batch.map((info, index) => `
          Video ${index + 1} (${info.videoFileName}):
          ---
          ${info.textContent}
          ---
        `).join('\n\n')}
      `;
      
      try {
        const result = await model.generateContent(batchPrompt);
        const responseText = result.response.text();
        
        // Parse the response
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          jsonStr = jsonMatch[1];
        }
        
        const parsedResponses = JSON.parse(jsonStr);
        
        // Add relevant videos to results
        parsedResponses.forEach((response, index) => {
          if (response.is_relevant && response.relevance_score > 0.5) {
            const videoInfo = batch[index];
            relevantVideos.push({
              filename: videoInfo.filename,
              videoPath: videoInfo.videoPath,
              videoFileName: videoInfo.videoFileName,
              processedAt: videoInfo.processedAt,
              score: response.relevance_score,
              justification: response.justification
            });
          }
        });
      } catch (error) {
        console.error(`Error evaluating batch ${i / BATCH_SIZE + 1}:`, error);
      }
    }
    
    // Sort by relevance score (descending)
    return relevantVideos.sort((a, b) => b.score - a.score);
  } catch (error) {
    console.error('Error searching video analyses:', error);
    throw error;
  }
}

// Function to handle chat conversation in video discussion
async function handleChatConversation(message, history, videoContext, memoryContext) {
  console.log('Handling chat conversation with message:', message);
  
  try {
    // Create a Gemini model instance
    const model = genAI.getGenerativeModel({ 
      model: 'gemini-1.5-flash',
      generationConfig: {
        temperature: 1,
        topP: 0.95,
        topK: 40,
        maxOutputTokens: 8192,
      }
    });
    
    // Format conversation history for Gemini
    const formattedHistory = history.map(msg => ({
      role: msg.type === 'user' ? 'user' : 'model',
      parts: [{ text: msg.content }]
    }));
    
    // Filter out system messages and ensure the first message is always from user
    let filteredHistory = formattedHistory.filter(msg => msg.role === 'user' || msg.role === 'model');
    
    // If history exists but doesn't start with user message, adjust accordingly
    if (filteredHistory.length > 0 && filteredHistory[0].role !== 'user') {
      filteredHistory = [];
    }
    
    // Start chat session with history if it exists
    const chatSession = model.startChat({
      history: filteredHistory.length >= 2 ? filteredHistory.slice(0, -1) : [],
    });
    
    // Prepare context information
    const contextInfo = `
You are an AI assistant helping a user discuss a video they've previously recorded. You have access to the following context:

VIDEO CONTEXT:
- Title: ${videoContext.videoFileName}
- Summary: ${videoContext.summary}
${videoContext.topics && videoContext.topics.length > 0 ? `- Topics: ${videoContext.topics.join(', ')}` : ''}
${videoContext.transcript ? '- Full transcript is available' : '- No transcript available'}

MEMORY CONTEXT:
- Working Memory: ${memoryContext.workingMemory?.established_facts?.length || 0} established facts and ${memoryContext.workingMemory?.untested_hypotheses?.length || 0} hypotheses
- Short-Term Memory: ${memoryContext.shortTermMemory?.length || 0} recent items
- Long-Term Memory: Profile information and knowledge base available

Use this context to provide informed, helpful responses about the video content and the user's memories related to it.
The user's message is: ${message}
`;

    // Send message with context
    const result = await chatSession.sendMessage(contextInfo);
    const response = result.response.text();
    
    console.log('Generated response:', response);
    return { response };
    
  } catch (error) {
    console.error('Error in chat conversation:', error);
    throw error;
  }
}

// New endpoint for searching video analyses
app.post('/api/search', async (req, res) => {
  try {
    if (!req.body.query) {
      return res.status(400).json({ error: 'Query parameter is required' });
    }
    
    const query = req.body.query;
    const results = await searchVideoAnalyses(query);
    
    res.json({ results });
  } catch (error) {
    console.error('Error searching videos:', error);
    res.status(500).json({ error: error.message || 'Failed to search videos' });
  }
});

// New endpoint for video discussion chat
app.post('/api/videos/chat', async (req, res) => {
  try {
    const { message, history, videoContext, memoryContext } = req.body;
    
    if (!message) {
      return res.status(400).json({ error: 'Message is required' });
    }
    
    if (!videoContext || !memoryContext) {
      return res.status(400).json({ error: 'Video and memory context are required' });
    }
    
    const result = await handleChatConversation(message, history || [], videoContext, memoryContext);
    res.json(result);
  } catch (error) {
    console.error('Error in chat endpoint:', error);
    res.status(500).json({ 
      error: error.message || 'Failed to process chat message',
      success: false 
    });
  }
});

// New endpoint for continuing discussion with a specific video context
app.post('/api/videos/continue-discussion', async (req, res) => {
  try {
    const { query, filename } = req.body;
    
    if (!query) {
      return res.status(400).json({ error: 'Query parameter is required' });
    }
    
    if (!filename) {
      return res.status(400).json({ error: 'Video filename is required' });
    }
    
    // Read the video file data
    const videoPath = path.join(DATASET_FOLDER, filename);
    
    // Check if file exists using fs.access instead of fs.existsSync
    try {
      await fs.access(videoPath);
    } catch (error) {
      return res.status(404).json({ error: 'Video data not found' });
    }
    
    const videoData = JSON.parse(await fs.readFile(videoPath, 'utf-8'));
    
    // Get current memory state
    const memoryState = await memoryManager.getMemoryState();
    
    // Prepare a response with the combined context
    const responseData = {
      success: true,
      videoContext: {
        videoFileName: videoData.videoFileName,
        processedAt: videoData.processedAt,
        summary: videoData.analysis?.summary || 'No summary available',
        transcript: videoData.analysis?.transcript || null,
        topics: videoData.analysis?.topics || [],
        insights: videoData.analysis?.inferred_insights || []
      },
      memoryContext: {
        workingMemory: memoryState.workingMemory,
        shortTermMemory: memoryState.shortTermMemory,
        longTermMemory: memoryState.longTermMemory
      },
      initialQuery: query
    };
    
    res.json(responseData);
  } catch (error) {
    console.error('Error continuing discussion:', error);
    res.status(500).json({ error: error.message || 'Failed to continue discussion' });
  }
});

// Endpoint to serve video files
app.get('/videos/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    const sourceVideoPath = path.join(WATCH_FOLDER, filename);
    
    // Check if file exists
    try {
      await fs.access(sourceVideoPath);
    } catch (error) {
      return res.status(404).send('Video file not found');
    }
    
    // Get file stats to determine size
    const stat = await fs.stat(sourceVideoPath);
    const fileSize = stat.size;
    const range = req.headers.range;
    
    // Handle range requests for video streaming
    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = (end - start) + 1;
      
      const fileStream = createReadStream(sourceVideoPath, { start, end });
      const head = {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': 'video/mp4',
      };
      
      res.writeHead(206, head);
      fileStream.pipe(res);
    } else {
      // No range requested, send entire file
      const head = {
        'Content-Length': fileSize,
        'Content-Type': 'video/mp4',
      };
      
      res.writeHead(200, head);
      createReadStream(sourceVideoPath).pipe(res);
    }
  } catch (error) {
    console.error('Error serving video:', error);
    res.status(500).send('Error serving video file');
  }
});

// New endpoint to serve thumbnails
app.get('/thumbnails/:filename', async (req, res) => {
  try {
    const filename = req.params.filename;
    const thumbnailPath = path.join(THUMBNAIL_FOLDER, filename);
    await fs.access(thumbnailPath);
    res.sendFile(thumbnailPath);
  } catch (error) {
    res.status(404).send('Thumbnail not found');
  }
});

// Initialize server
const port = process.env.PORT || 8001;
const server = ViteExpress.listen(app, port, () => {
  console.log(`Server listening on port ${port}`);
  console.log(`Watching folder: ${WATCH_FOLDER}`);
  console.log(`Dataset folder: ${DATASET_FOLDER}`);
});

setupWatcher().catch(error => {
  console.error('Failed to set up file watcher:', error);
}); 


================================================
File: video-watcher/server/memory-manager.mjs
================================================
import fs from 'fs/promises';
import path from 'path';
import { GoogleGenerativeAI } from '@google/generative-ai';

// Constants for memory management
const MEMORY_DIR = path.join(process.cwd(), 'memory');
const STM_TOKEN_LIMIT = 8000;
const LTM_TOKEN_LIMIT = 8000;
const WM_TOKEN_LIMIT = 8000;
const DEFAULT_MODEL = 'gemini-2.0-flash';
const SUMMARY_MODEL = 'gemini-2.5-pro-exp-03-25'
const MEMORY_STATE_FILE = path.join(MEMORY_DIR, 'memory-state.json');

// Initialize Gemini AI
const genAI = new GoogleGenerativeAI(process.env.VITE_GEMINI_API_KEY);

/**
 * Multi-tiered memory system manager
 */
export class MemoryManager {
  constructor() {
    this.shortTermMemory = []; // In-memory array of recent events
    this.longTermMemory = {}; // Structured persistent memory
    this.workingMemory = {
      untested_hypotheses: [],
      corroborated_hypotheses: [],
      established_facts: []
    };
    this.initialized = false;
  }

  /**
   * Initialize the memory manager by loading persisted state
   */
  async initialize() {
    try {
      // Ensure memory directory exists
      await fs.mkdir(MEMORY_DIR, { recursive: true });
      
      // Try to load the complete memory state first
      try {
        const stateContent = await fs.readFile(MEMORY_STATE_FILE, 'utf-8');
        const savedState = JSON.parse(stateContent);
        
        // Restore complete state
        this.shortTermMemory = savedState.shortTermMemory || [];
        this.longTermMemory = savedState.longTermMemory || {};
        this.workingMemory = savedState.workingMemory || {
          untested_hypotheses: [],
          corroborated_hypotheses: [],
          established_facts: []
        };
        
        console.log('Loaded complete memory state from memory-state.json');
      } catch (error) {
        console.log('No existing complete memory state found, trying individual memory files');
        
        // Fall back to loading individual memory files if complete state isn't available
        try {
          const ltmContent = await fs.readFile(path.join(MEMORY_DIR, 'ltm.json'), 'utf-8');
          this.longTermMemory = JSON.parse(ltmContent);
          console.log('Loaded long-term memory');
        } catch (error) {
          console.log('No existing LTM found, initializing empty LTM');
          this.longTermMemory = {};
        }
        
        try {
          const wmContent = await fs.readFile(path.join(MEMORY_DIR, 'wm.json'), 'utf-8');
          this.workingMemory = JSON.parse(wmContent);
          console.log('Loaded working memory');
        } catch (error) {
          console.log('No existing WM found, initializing empty WM');
        }
        
        try {
          const stmContent = await fs.readFile(path.join(MEMORY_DIR, 'stm.json'), 'utf-8');
          this.shortTermMemory = JSON.parse(stmContent);
          console.log('Loaded short-term memory');
        } catch (error) {
          console.log('No existing STM found, initializing empty STM');
        }
        
        // Save the initial state to create the complete state file
        await this.persistCompleteState();
      }
      
      this.initialized = true;
      console.log('Memory Manager initialized');
    } catch (error) {
      console.error('Error initializing memory manager:', error);
      throw error;
    }
  }

  /**
   * Estimate token count for a string (rough approximation)
   * @param {string} text - Text to estimate token count for
   * @returns {number} - Estimated token count
   */
  estimateTokens(text) {
    // Simple estimation: 1 token ~ 4 characters
    return Math.ceil(text.length / 4);
  }

  /**
   * Add new entries to short-term memory and trigger processing
   * @param {object} analysisResult - Result of video analysis including explicit directives and inferred insights
   */
  async processNewAnalysis(analysisResult) {
    if (!this.initialized) {
      await this.initialize();
    }

    const timestamp = new Date().toISOString();
    const contextEntry = {
      timestamp,
      type: 'video_analysis_summary',
      data: {
        summary: analysisResult.relevantContextSummary || analysisResult.summary || "No summary available"
      }
    };
    
    // Add context summary as a single entry
    this.shortTermMemory.push(contextEntry);
    
    // Add each explicit directive as a separate entry
    if (analysisResult.explicit_directives && Array.isArray(analysisResult.explicit_directives)) {
      for (const directive of analysisResult.explicit_directives) {
        this.shortTermMemory.push({
          timestamp,
          type: 'explicit_directive',
          data: directive
        });
      }
    }
    
    // Add each explicit statement as a separate entry
    if (analysisResult.explicit_statements && Array.isArray(analysisResult.explicit_statements)) {
      for (const statement of analysisResult.explicit_statements) {
        this.shortTermMemory.push({
          timestamp,
          type: 'explicit_statement',
          data: statement
        });
      }
    }
    
    // Add each inferred insight as a separate entry
    if (analysisResult.inferred_insights && Array.isArray(analysisResult.inferred_insights)) {
      for (const insight of analysisResult.inferred_insights) {
        this.shortTermMemory.push({
          timestamp,
          type: 'inferred_insight',
          data: insight
        });
      }
    }
    
    console.log(`Added ${1 + 
      (analysisResult.explicit_directives?.length || 0) + 
      (analysisResult.explicit_statements?.length || 0) + 
      (analysisResult.inferred_insights?.length || 0)} entries to STM`);
    
    // Persist STM after adding new entries
    await this.persistSTM();
    
    // Save complete memory state
    await this.persistCompleteState();
    
    // Check if STM needs consolidation
    await this.checkSTMSize();
    
    // Update working memory with new insights
    await this.updateWorkingMemory();
  }

  /**
   * Check if STM exceeds token limit and consolidate if needed
   */
  async checkSTMSize() {
    try {
      // Estimate total tokens in STM
      const stmString = JSON.stringify(this.shortTermMemory);
      const tokenCount = this.estimateTokens(stmString);
      
      console.log(`STM size: ${tokenCount} tokens (limit: ${STM_TOKEN_LIMIT})`);
      
      // If STM exceeds token limit, consolidate oldest entries into LTM
      if (tokenCount > STM_TOKEN_LIMIT) {
        console.log('STM exceeds token limit, consolidating to LTM...');
        await this.consolidateToLTM();
      }
    } catch (error) {
      console.error('Error checking STM size:', error);
    }
  }

  /**
   * Consolidate oldest STM entries into LTM
   */
  async consolidateToLTM() {
    try {
      // Sort entries by timestamp
      this.shortTermMemory.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
      
      // Take oldest ~3000 tokens worth of entries
      let oldestEntries = [];
      let tokenCount = 0;
      let i = 0;
      
      while (i < this.shortTermMemory.length && tokenCount < 3000) {
        oldestEntries.push(this.shortTermMemory[i]);
        tokenCount += this.estimateTokens(JSON.stringify(this.shortTermMemory[i]));
        i++;
      }
      
      if (oldestEntries.length === 0) {
        console.log('No entries to consolidate');
        return;
      }
      
      console.log(`Consolidating ${oldestEntries.length} oldest entries (${tokenCount} tokens) to LTM`);
      
      // Create LTM summary using Gemini
      const updatedLTM = await this.createLTMSummary(oldestEntries);
      
      // Update LTM with new summary
      this.longTermMemory = updatedLTM;
      
      // Check LTM size and trim if needed
      await this.checkLTMSize();
      
      // Persist updated LTM
      await this.persistLTM();
      
      // Remove consolidated entries from STM
      this.shortTermMemory.splice(0, oldestEntries.length);
      
      // Persist updated STM
      await this.persistSTM();
      
      // Save complete memory state
      await this.persistCompleteState();
      
      console.log(`STM consolidated. ${this.shortTermMemory.length} entries remaining.`);
    } catch (error) {
      console.error('Error consolidating to LTM:', error);
    }
  }

  /**
   * Check and trim LTM if it exceeds token limit
   */
  async checkLTMSize() {
    try {
      const ltmString = JSON.stringify(this.longTermMemory);
      const tokenCount = this.estimateTokens(ltmString);
      
      console.log(`LTM size: ${tokenCount} tokens (limit: ${LTM_TOKEN_LIMIT})`);
      
      if (tokenCount > LTM_TOKEN_LIMIT) {
        console.log('LTM exceeds token limit, trimming...');
        await this.trimLTM(tokenCount);
      }
    } catch (error) {
      console.error('Error checking LTM size:', error);
    }
  }

  /**
   * Trim LTM to stay within token limit
   */
  async trimLTM(currentTokenCount) {
    try {
      // Use Gemini to create a more concise summary
      const prompt = `
The following is the current long-term memory for a user assistant that exceeds our token limit of ${LTM_TOKEN_LIMIT}.
Current size: approximately ${currentTokenCount} tokens.

Please condense this information to a more concise representation while preserving the most important insights.
Focus on:
1. Core user preferences and established patterns
2. Most relevant skills, knowledge, and workflows
3. Highest confidence insights and explicitly stated facts

Current LTM:
---
${JSON.stringify(this.longTermMemory, null, 2)}
---

Return a condensed version in the same JSON structure, but more concise and within our ${LTM_TOKEN_LIMIT} token limit.
Ensure the output is a valid JSON object with the same structure.`;

      const model = genAI.getGenerativeModel({ model: SUMMARY_MODEL });
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      
      try {
        // Extract JSON from response
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          jsonStr = jsonMatch[1];
        }
        
        const trimmedLTM = JSON.parse(jsonStr);
        const newTokenCount = this.estimateTokens(JSON.stringify(trimmedLTM));
        
        if (newTokenCount <= LTM_TOKEN_LIMIT) {
          this.longTermMemory = trimmedLTM;
          console.log(`Successfully trimmed LTM to ${newTokenCount} tokens`);
        } else {
          console.warn(`Trimmed LTM still exceeds token limit (${newTokenCount} tokens)`);
          // Force additional trimming by removing less important categories
          const priorityOrder = [
            "profile_summary", 
            "skills_and_knowledge.confirmed_skills",
            "preferences_and_habits.ui_preferences",
            "preferences_and_habits.tool_preferences",
            "goals_and_motivations.stated_goals",
            "challenges.recurring_frustrations"
          ];
          
          // Keep trimming based on priority until we're under the limit
          this.longTermMemory = this.forceTrimByPriority(trimmedLTM, priorityOrder);
        }
      } catch (parseError) {
        console.error('Error parsing trimmed LTM JSON:', parseError);
        // If parsing fails, we'll do a manual basic trimming
        this.longTermMemory = this.basicTrimLTM();
      }
    } catch (error) {
      console.error('Error trimming LTM:', error);
      // Fallback to basic trimming
      this.longTermMemory = this.basicTrimLTM();
    }
  }

  /**
   * Force trim LTM by keeping high priority elements
   */
  forceTrimByPriority(ltm, priorityOrder) {
    // Deep clone to avoid modifying the original
    const result = JSON.parse(JSON.stringify(ltm));
    
    // Start with only the prioritized fields
    const trimmed = {};
    priorityOrder.forEach(path => {
      const parts = path.split('.');
      if (parts.length === 1) {
        if (result[parts[0]]) {
          if (!trimmed[parts[0]]) trimmed[parts[0]] = result[parts[0]];
        }
      } else if (parts.length === 2) {
        if (!trimmed[parts[0]]) trimmed[parts[0]] = {};
        if (result[parts[0]] && result[parts[0]][parts[1]]) {
          trimmed[parts[0]][parts[1]] = result[parts[0]][parts[1]]; 
        }
      }
    });
    
    console.log('Created priority-based trimmed LTM');
    return trimmed;
  }

  /**
   * Basic trim LTM as a fallback
   */
  basicTrimLTM() {
    // Simple fallback - keep only profile summary and most important categories
    const basic = {
      profile_summary: this.longTermMemory.profile_summary || "User profile",
      skills_and_knowledge: {
        confirmed_skills: this.longTermMemory.skills_and_knowledge?.confirmed_skills?.slice(0, 5) || []
      },
      preferences_and_habits: {
        ui_preferences: this.longTermMemory.preferences_and_habits?.ui_preferences?.slice(0, 5) || []
      }
    };
    
    console.log('Created basic trimmed LTM due to errors in advanced trimming');
    return basic;
  }

  /**
   * Update working memory based on STM and LTM
   */
  async updateWorkingMemory() {
    try {
      // Only update if we have STM entries
      if (this.shortTermMemory.length === 0) {
        console.log('No STM entries to update working memory');
        return;
      }
      
      console.log('Updating working memory...');
      
      // Format STM entries for the prompt
      const recentSTM = this.shortTermMemory.slice(-20); // Take most recent entries for context
      const formattedSTM = recentSTM.map(entry => {
        return `[${entry.timestamp}] (${entry.type}): ${JSON.stringify(entry.data)}`;
      }).join('\n');
      
      // Improved working memory reasoning prompt
      const prompt = `
You are an advanced cognitive model that builds a coherent user mental model by analyzing recent activity, long-term patterns, and current context.

Your task is to update the Working Memory (WM) to reflect the user's current state, goals, needs, and context.

Current WM:
---
${JSON.stringify(this.workingMemory, null, 2)}
---

STM (Recent Activity & Inferences):
---
${formattedSTM}
---

LTM (Long-Term Profile):
---
${JSON.stringify(this.longTermMemory, null, 2)}
---

INSTRUCTIONS:

1. ANALYZE recent STM entries through the lens of existing LTM and current WM.

2. Maintain these three categories in WM:
   a) UNTESTED HYPOTHESES: Fresh observations that seem plausible but need more evidence
   b) CORROBORATED HYPOTHESES: Observations with moderate support across multiple interactions
   c) ESTABLISHED FACTS: Consistently supported observations or explicitly stated information

3. For each hypothesis/fact, include:
   - The specific insight written concisely but precisely
   - The evidence basis in [brackets]
   - Relevance to the user's current context/goals
   
4. Focus on what would help understand and assist the user RIGHT NOW.

5. Maintain cognitive hierarchy:
   - PROMOTE untested hypotheses to corroborated when additional evidence appears
   - PROMOTE corroborated hypotheses to facts when consistently supported
   - DEMOTE or REMOVE when evidence contradicts

The updated WM should prioritize insights that are:
- Actionable (can inform immediate recommendations)
- Context-aware (relevant to current session)
- Specific (detailed enough to guide decisions)
- Evidence-based (clearly linked to observations)

Output the updated WM as a JSON object with these three arrays. Ensure the total response stays within ${WM_TOKEN_LIMIT} tokens.
`;

      // Generate updated working memory using Gemini
      const model = genAI.getGenerativeModel({ model: DEFAULT_MODEL });
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      
      try {
        // Extract JSON from response (handles both raw JSON or markdown-wrapped JSON)
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          jsonStr = jsonMatch[1];
        }
        
        const updatedWM = JSON.parse(jsonStr);
        this.workingMemory = updatedWM;
        console.log('Successfully updated working memory');
        
        // Check WM size and trim if needed
        await this.checkWMSize();
        
        // Persist updated WM
        await this.persistWM();
        
        // Save complete memory state
        await this.persistCompleteState();
      } catch (parseError) {
        console.error('Error parsing working memory JSON:', parseError);
        console.log('Raw WM response:', responseText);
      }
    } catch (error) {
      console.error('Error updating working memory:', error);
    }
  }

  /**
   * Check and trim WM if it exceeds token limit
   */
  async checkWMSize() {
    try {
      const wmString = JSON.stringify(this.workingMemory);
      const tokenCount = this.estimateTokens(wmString);
      
      console.log(`WM size: ${tokenCount} tokens (limit: ${WM_TOKEN_LIMIT})`);
      
      if (tokenCount > WM_TOKEN_LIMIT) {
        console.log('WM exceeds token limit, trimming...');
        await this.trimWM();
      }
    } catch (error) {
      console.error('Error checking WM size:', error);
    }
  }

  /**
   * Trim WM to stay within token limit
   */
  async trimWM() {
    try {
      // First try to trim by priority
      const trimmed = { ...this.workingMemory };
      
      // Keep all established facts
      // Reduce corroborated hypotheses if needed
      if (trimmed.corroborated_hypotheses && trimmed.corroborated_hypotheses.length > 10) {
        trimmed.corroborated_hypotheses = trimmed.corroborated_hypotheses.slice(0, 10);
      }
      
      // Reduce untested hypotheses more aggressively
      if (trimmed.untested_hypotheses && trimmed.untested_hypotheses.length > 5) {
        trimmed.untested_hypotheses = trimmed.untested_hypotheses.slice(0, 5);
      }
      
      const tokenCount = this.estimateTokens(JSON.stringify(trimmed));
      if (tokenCount <= WM_TOKEN_LIMIT) {
        this.workingMemory = trimmed;
        console.log(`Trimmed WM to ${tokenCount} tokens by reducing hypotheses count`);
        return;
      }
      
      // If still too large, use Gemini to create a more concise version
      const prompt = `
The following working memory for a user assistant exceeds our token limit of ${WM_TOKEN_LIMIT}.

Please condense this working memory while preserving the most important insights.
Focus on:
1. All established facts
2. Most relevant corroborated hypotheses
3. Only the most recent and actionable untested hypotheses

Current WM:
---
${JSON.stringify(this.workingMemory, null, 2)}
---

Return a condensed version with the same structure but more concise entries.
Make sure to maintain the three categories: untested_hypotheses, corroborated_hypotheses, and established_facts.
Ensure the output is a valid JSON object and stays within ${WM_TOKEN_LIMIT} tokens.`;

      const model = genAI.getGenerativeModel({ model: DEFAULT_MODEL });
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      
      try {
        // Extract JSON from response
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          jsonStr = jsonMatch[1];
        }
        
        const trimmedWM = JSON.parse(jsonStr);
        this.workingMemory = trimmedWM;
        console.log('Successfully trimmed WM using Gemini');
      } catch (parseError) {
        console.error('Error parsing trimmed WM JSON:', parseError);
        // If parsing fails, revert to the basic trimming we tried earlier
        this.workingMemory = trimmed;
      }
    } catch (error) {
      console.error('Error trimming WM:', error);
    }
  }

  /**
   * Persist long-term memory to disk
   */
  async persistLTM() {
    try {
      await fs.writeFile(
        path.join(MEMORY_DIR, 'ltm.json'),
        JSON.stringify(this.longTermMemory, null, 2),
        'utf-8'
      );
      console.log('Persisted LTM to disk');
    } catch (error) {
      console.error('Error persisting LTM:', error);
    }
  }

  /**
   * Persist working memory to disk
   */
  async persistWM() {
    try {
      await fs.writeFile(
        path.join(MEMORY_DIR, 'wm.json'),
        JSON.stringify(this.workingMemory, null, 2),
        'utf-8'
      );
      console.log('Persisted WM to disk');
    } catch (error) {
      console.error('Error persisting WM:', error);
    }
  }

  /**
   * Persist short-term memory to disk
   */
  async persistSTM() {
    try {
      await fs.writeFile(
        path.join(MEMORY_DIR, 'stm.json'),
        JSON.stringify(this.shortTermMemory, null, 2),
        'utf-8'
      );
      console.log('Persisted STM to disk');
    } catch (error) {
      console.error('Error persisting STM:', error);
    }
  }

  /**
   * Persist the complete memory state to a single JSON file
   */
  async persistCompleteState() {
    try {
      const completeState = {
        shortTermMemory: this.shortTermMemory,
        longTermMemory: this.longTermMemory,
        workingMemory: this.workingMemory,
        lastUpdated: new Date().toISOString()
      };
      
      await fs.writeFile(
        MEMORY_STATE_FILE,
        JSON.stringify(completeState, null, 2),
        'utf-8'
      );
      console.log('Persisted complete memory state to disk');
    } catch (error) {
      console.error('Error persisting complete memory state:', error);
    }
  }

  /**
   * Get the current memory state
   * @returns {Object} - Current memory state
   */
  getMemoryState() {
    return {
      shortTermMemory: this.shortTermMemory,
      longTermMemory: this.longTermMemory,
      workingMemory: this.workingMemory
    };
  }

  /**
   * Create LTM summary from STM entries using Gemini
   * @param {Array} stmEntries - STM entries to summarize
   * @returns {Object} - Updated LTM object
   */
  async createLTMSummary(stmEntries) {
    try {
      // Format STM entries for the prompt
      const formattedSTMEntries = stmEntries.map(entry => {
        return `[${entry.timestamp}] (${entry.type}): ${JSON.stringify(entry.data)}`;
      }).join("\n");
      
      // Improved LTM summarization prompt
      const prompt = `
You are an advanced cognitive system that builds and maintains a rich user profile from interaction history.

Your task: Synthesize new observations into the user's Long-Term Memory (LTM) profile, integrating them with existing knowledge.

EXISTING LTM:
---
${JSON.stringify(this.longTermMemory, null, 2)}
---

NEW OBSERVATIONS TO INTEGRATE:
---
${formattedSTMEntries}
---

Follow these guidelines:

1. MERGE observations with the existing LTM, REFINING understanding rather than just adding items
2. PRIORITIZE explicit statements over inferences when they conflict
3. INDICATE confidence levels for inferred traits (high/medium/low)
4. FOCUS on patterns that reveal:
   - Skill proficiency and knowledge areas
   - UI/UX preferences and workflow habits
   - Recurring frustrations and challenges
   - Goals and motivations driving behavior
   - Communication and learning style

5. CONDENSE redundant or similar entries to maintain a clean profile
6. REMOVE outdated information when new evidence suggests a change
7. STRUCTURE the profile hierarchically using the template below

Output the ENTIRE UPDATED LTM as a valid JSON object with this structure:
{
  "profile_summary": "Brief overview of user's primary traits and patterns",
  "skills_and_knowledge": {
    "confirmed_skills": [...],
    "inferred_skills": [...],
    "knowledge_gaps": [...]
  },
  "preferences_and_habits": {
    "ui_preferences": [...],
    "workflow_habits": [...],
    "tool_preferences": [...]
  },
  "workflows": {
    "common_tasks": [...],
    "approaches": [...],
    "frequency_patterns": [...]
  },
  "challenges": {
    "recurring_frustrations": [...],
    "difficulties": [...],
    "blockers": [...]
  },
  "goals_and_motivations": {
    "stated_goals": [...],
    "inferred_goals": [...],
    "motivations": [...]
  },
  "traits_and_attitudes": {
    "communication_style": [...],
    "decision_making": [...],
    "learning_approach": [...]
  }
}

Ensure the output stays within approximately ${LTM_TOKEN_LIMIT} tokens and is valid JSON without trailing commas.`;

      // Generate LTM summary using Gemini
      const model = genAI.getGenerativeModel({ model: SUMMARY_MODEL });
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      
      try {
        // Extract JSON from response (handles both raw JSON or markdown-wrapped JSON)
        let jsonStr = responseText;
        const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
        if (jsonMatch && jsonMatch[1]) {
          jsonStr = jsonMatch[1];
        }
        
        const updatedLTM = JSON.parse(jsonStr);
        console.log('Successfully created LTM summary');
        return updatedLTM;
      } catch (parseError) {
        console.error('Error parsing LTM summary JSON:', parseError);
        // If parsing fails, keep existing LTM and log the error
        console.log('Raw LTM response:', responseText);
        return this.longTermMemory;
      }
    } catch (error) {
      console.error('Error creating LTM summary:', error);
      return this.longTermMemory;
    }
  }

  /**
   * Create a conversational interface to the memory system
   * @param {string} query - User query about the memory system
   * @returns {string} - Response from the memory system
   */
  async conversationalMemoryQuery(query) {
    try {
      if (!this.initialized) {
        await this.initialize();
      }

      console.log(`Processing memory query: ${query}`);
      
      // Format current memory state for the prompt
      const memoryState = {
        stm: this.shortTermMemory.slice(-10), // Last 10 STM entries for context
        ltm: this.longTermMemory,
        wm: this.workingMemory
      };
      
      const prompt = `
You are the Memory Portal, an interface to a multi-tiered memory system for an AI assistant.
The system includes:

1. Short-Term Memory (STM): Recent observations and inferences
2. Long-Term Memory (LTM): Persistent user profile and patterns
3. Working Memory (WM): Current hypotheses and established facts

The user is asking about this memory system. Respond helpfully, transparently, and concisely.

Current Memory State:
---
${JSON.stringify(memoryState, null, 2)}
---

User Query: "${query}"

Guidelines:
- If the query is about the CONTENT of memories, answer based on the data shown above
- If the query is about EDITING memories, explain how memories are processed and consolidated
- If the query is about HOW THE SYSTEM WORKS, explain the relevant components
- If the query is a COMMAND to update memory, respond as if you've made the change (the actual implementation will happen elsewhere)
- Keep your response concise but informative
- Be transparent about confidence levels when discussing inferences vs. explicit observations

Remember your role as a Memory Portal - you provide access to the system's knowledge about the user, not general knowledge.`;

      // Generate response using Gemini
      const model = genAI.getGenerativeModel({ model: SUMMARY_MODEL });
      const result = await model.generateContent(prompt);
      const responseText = result.response.text();
      
      console.log('Generated memory portal response');
      return responseText;
    } catch (error) {
      console.error('Error in conversational memory query:', error);
      return "Sorry, I encountered an error accessing the memory system. Please try again.";
    }
  }
}

// Export singleton instance
const memoryManager = new MemoryManager();
export default memoryManager; 


================================================
File: video-watcher/server/video-processor.mjs
================================================
import fs from 'fs/promises';
import path from 'path';
import { GoogleGenerativeAI } from '@google/generative-ai';
import { GoogleAIFileManager } from '@google/generative-ai/server';
import { exec } from 'child_process';

// Initialize Gemini AI - using the same approach as video folder project
const key = process.env.VITE_GEMINI_API_KEY;
const fileManager = new GoogleAIFileManager(key);
const genAI = new GoogleGenerativeAI(key);

// Export genAI for use in other modules
export { genAI };

// Default model to use if not specified
const DEFAULT_MODEL = 'gemini-2.0-flash';

// Prompt template for video analysis
const DEFAULT_PROMPT = `
Analyze this video recording and provide a detailed description of:
1. The content visible on the screen
2. Any actions or activities being performed
3. Key topics discussed or shown
4. Transcribe any spoken content

Structure your response as a JSON object with the following fields:
{
  "summary": "Detailed summary of the video",
  "screenContent": "Description of what's visible on the screen",
  "actions": "Description of actions performed",
  "topics": ["topic1", "topic2"],
  "transcript": "Transcription of speech",
  "tags": ["tag1", "tag2"]
}
`;

// Prompt template for insight inference directly from video
const INFERENCE_PROMPT = `
Analyze this screen recording to identify both explicit statements and infer the user's underlying mental state, intentions, and tacit knowledge. Go beyond the literal content to provide deeper insights.

Focus on:
1.  **Explicit Directives & Preferences:** Commands (implicit/explicit) for an AI, stated preferences, goals, or expressed frustrations.
2.  **Inferred User State & Intentions:** Interpret the user's actions and words to deduce:
    *   **Mental/Emotional State:** Signs of confusion, focus, frustration, satisfaction, contemplation, cognitive load, etc. (e.g., "User seems hesitant", "User sounds frustrated with the loading time").
    *   **Underlying Goals/Motivations:** What is the user *really* trying to achieve, even if not stated? (e.g., "Appears to be learning [Software X]", "Trying to optimize their workflow for task Y").
    *   **Unspoken Needs/Desires:** What might the user want or need based on their actions? (e.g., "User seems to be looking for a shortcut", "Might benefit from an explanation of [Concept Z]").
    *   **Observed Workflow/Habits:** Patterns in how the user interacts with the system or performs tasks (e.g., "Prefers using keyboard shortcuts", "Methodically checks settings before proceeding", "Often multi-tasks between App A and App B").
    *   **Potential Knowledge Gaps:** Areas where the user seems uncertain or lacks information (e.g., "Unsure how feature X works", "Searching for basic commands").
    *   **Implied Opinions/Critiques:** Subtle judgments about tools, processes, or outcomes, even if not voiced directly (e.g., "Seems unimpressed by the tool's speed", "Appears to implicitly prefer Tool A over Tool B for this task").
    *   **Withheld Recommendations/Ideas:** Potential improvements or alternative approaches the user might be considering but not stating (e.g., "Might be thinking about automating this step", "Considering a different tool for the next step").

Guidelines for analysis:
1. Focus on making reasonable inferences based on evidence in the video
2. Include a basis or reasoning for each inference to explain your thinking
3. Rate your certainty for each insight (high/medium/low)
4. Be specific and actionable rather than vague
5. Look for patterns in the user's behavior, language, and screen interactions

Structure the response as a JSON object:
{
  "explicit_directives": [
    { 
      "command": "The specific instruction or command detected", 
      "target": "What/who the command is directed to", 
      "parameters": {"param1": "value1"}, 
      "certainty": "high/medium/low", 
      "context": "Description of when/how this directive was given" 
    }
  ],
  "explicit_statements": [
    { 
      "statement": "The explicit statement made by the user", 
      "type": "preference/goal/frustration/interest/question", 
      "certainty": "high/medium/low", 
      "context": "Description of when/how this statement was made" 
    }
  ],
  "inferred_insights": [
    { 
      "insight": "The inferred insight about the user's state, goals, needs, etc.", 
      "type": "mental_state/goal/need/workflow/knowledge_gap/opinion/withheld_idea", 
      "basis": "The specific observation or pattern that led to this inference", 
      "certainty": "high/medium/low" 
    }
  ],
  "relevant_context_summary": "Brief summary of the video focusing on aspects most relevant to understanding the user's current state and goals."
}

If no significant explicit items or inferences can be made, return empty arrays for the respective fields but provide the context summary. Be specific in the 'basis' field for inferences.
`;

/**
 * Wait for a specified time
 * @param {number} ms - Time to wait in milliseconds
 * @returns {Promise<void>}
 */
const wait = (ms) => new Promise(resolve => setTimeout(resolve, ms));

/**
 * Checks the progress of file processing
 * @param {string} fileId - ID of the uploaded file
 * @returns {Promise<Object>} - File status information
 */
async function checkProgress(fileId) {
  try {
    const result = await fileManager.getFile(fileId);
    return result;
  } catch (error) {
    console.error('Error checking file progress:', error);
    return { error };
  }
}

/**
 * Uploads a video file to Google AI for processing
 * @param {string} filePath - Path to the video file
 * @returns {Promise<Object>} - Upload result
 */
async function uploadVideoFile(filePath) {
  try {
    const fileName = path.basename(filePath);
    const fileStats = await fs.stat(filePath);
    console.log(`Uploading ${fileName} (${fileStats.size} bytes)...`);
    
    const uploadResult = await fileManager.uploadFile(filePath, {
      displayName: fileName,
      mimeType: 'video/mp4'
    });
    
    console.log(`Upload successful: ${uploadResult.file.name}`);
    return uploadResult.file;
  } catch (error) {
    console.error('Error uploading video:', error);
    throw error;
  }
}

/**
 * Analyzes a video using Google Gemini
 * @param {string} filePath - Path to the video file
 * @param {string} customPrompt - Optional custom prompt
 * @returns {Promise<Object>} - Analysis results
 */
export async function analyzeVideo(filePath, customPrompt = DEFAULT_PROMPT) {
  try {
    // Upload the video file
    const uploadResult = await uploadVideoFile(filePath);
    
    // Wait for file processing (checking progress)
    console.log(`Checking progress for file ${uploadResult.name}...`);
    let isReady = false;
    let retryCount = 0;
    const baseWaitTime = 2000; // Start with 2 second wait
    
    while (!isReady) {
      const progress = await checkProgress(uploadResult.name);
      console.log(`File status: ${JSON.stringify(progress)}`);
      
      if (progress.state === 'ACTIVE') {
        isReady = true;
        break;
      }
      
      // Calculate wait time with exponential backoff, capped at 30 seconds
      const waitTime = Math.min(baseWaitTime * Math.pow(1.5, retryCount), 30000);
      console.log(`File not ready, waiting ${waitTime/1000}s before retry...`);
      await wait(waitTime);
      retryCount++;
    }
    
    // Create request for Gemini for basic video analysis
    const req = [
      { text: customPrompt },
      {
        fileData: {
          mimeType: uploadResult.mimeType,
          fileUri: uploadResult.uri
        }
      }
    ];
    
    console.log(`Sending to Gemini (model: ${DEFAULT_MODEL}) for basic analysis...`);
    const result = await genAI.getGenerativeModel({ model: DEFAULT_MODEL }).generateContent(req);
    
    console.log('Response received from Gemini for basic analysis');
    const responseText = result.response.text();
    
    // Try to parse the JSON response
    try {
      // The response might have markdown formatting with JSON inside ```json blocks
      let jsonStr = responseText;
      
      // Check if response is wrapped in markdown code blocks
      const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (jsonMatch && jsonMatch[1]) {
        jsonStr = jsonMatch[1];
      }
      
      const parsedResponse = JSON.parse(jsonStr);
      
      // Make a second API call for deeper insights using the same video file
      console.log('Making second API call for deeper inference analysis...');
      const inferenceResult = await analyzeVideoForInsights(uploadResult);
      
      // Combine both results
      const enhancedResponse = {
        ...parsedResponse,
        ...inferenceResult
      };
      
      return {
        ...enhancedResponse,
        text: responseText,
        candidates: result.response.candidates,
        feedback: result.response.promptFeedback
      };
    } catch (parseError) {
      console.warn('Could not parse response as JSON, returning raw text');
      return {
        rawResponse: responseText,
        text: responseText,
        candidates: result.response.candidates,
        feedback: result.response.promptFeedback,
        error: 'Response could not be parsed as JSON'
      };
    }
  } catch (error) {
    console.error('Error analyzing video:', error);
    
    // Return a simplified response with the error for testing
    return {
      error: error.message,
      summary: "Error processing video with Gemini",
      screenContent: "Could not analyze screen content due to API error",
      actions: "No actions detected due to processing error",
      topics: ["error", "processing failed"],
      transcript: "Transcript unavailable due to processing error",
      tags: ["error", "api-failure", "test-data"]
    };
  }
}

/**
 * Analyzes a video to infer explicit directives and deeper insights
 * @param {Object} uploadedFile - The uploaded file object from Gemini API
 * @returns {Promise<Object>} - Analysis results with explicit and inferred information
 */
async function analyzeVideoForInsights(uploadedFile) {
  try {
    console.log('Analyzing video for explicit directives and inferred insights...');
    
    // Create request for Gemini with inference prompt
    const req = [
      { text: INFERENCE_PROMPT },
      {
        fileData: {
          mimeType: uploadedFile.mimeType,
          fileUri: uploadedFile.uri
        }
      }
    ];
    
    // Use the same model for both analyses
    const model = genAI.getGenerativeModel({ model: DEFAULT_MODEL });
    
    // Generate inference analysis
    console.log(`Sending to Gemini (model: ${DEFAULT_MODEL}) for inference analysis...`);
    const result = await model.generateContent(req);
    const responseText = result.response.text();
    
    try {
      // Try to parse the JSON response (handling both raw JSON and markdown-wrapped JSON)
      let jsonStr = responseText;
      const jsonMatch = responseText.match(/```(?:json)?\s*([\s\S]*?)\s*```/);
      if (jsonMatch && jsonMatch[1]) {
        jsonStr = jsonMatch[1];
      }
      
      const inferenceResult = JSON.parse(jsonStr);
      console.log('Successfully parsed inference analysis');
      
      return inferenceResult;
    } catch (parseError) {
      console.error('Error parsing inference analysis:', parseError);
      console.log('Raw inference response:', responseText);
      
      // Return default structure with empty arrays if parsing fails
      return {
        explicit_directives: [],
        explicit_statements: [],
        inferred_insights: [],
        relevant_context_summary: "Error parsing inference analysis"
      };
    }
  } catch (error) {
    console.error('Error analyzing video for insights:', error);
    
    // Return default structure with empty arrays if analysis fails
    return {
      explicit_directives: [],
      explicit_statements: [],
      inferred_insights: [],
      relevant_context_summary: `Error analyzing video: ${error.message}`
    };
  }
}

/**
 * Saves analysis results to the dataset
 * @param {string} videoPath - Path to the original video
 * @param {Object} analysisResult - Analysis results from Gemini
 * @param {string} datasetFolder - Path to the dataset folder
 */
export async function saveToDataset(videoPath, analysisResult, datasetFolder) {
  try {
    const videoFileName = path.basename(videoPath);
    const timestamp = new Date().toISOString();
    
    // Create a dataset entry
    const datasetEntry = {
      id: `video_${Date.now()}`,
      videoFileName: videoFileName,
      videoPath: videoPath,
      processedAt: timestamp,
      analysis: analysisResult,
      // Include inferred insights in the dataset entry if available
      inferred_insights: analysisResult.inferred_insights || []
    };
    
    // Save to the dataset folder
    const jsonFileName = `${path.parse(videoFileName).name}.json`;
    const jsonPath = path.join(datasetFolder, jsonFileName);
    
    await fs.writeFile(
      jsonPath, 
      JSON.stringify(datasetEntry, null, 2), 
      'utf-8'
    );
    
    return {
      success: true,
      datasetPath: jsonPath
    };
  } catch (error) {
    console.error('Error saving to dataset:', error);
    throw new Error(`Failed to save to dataset: ${error.message}`);
  }
}

/**
 * Generate a thumbnail at 5 seconds into the video using ffmpeg
 * @param {string} videoPath - Path to the video file
 * @param {string} thumbnailPath - Path to save the generated thumbnail image
 * @returns {Promise<void>} Resolves when the thumbnail is created
 */
export function generateThumbnail(videoPath, thumbnailPath) {
  return new Promise((resolve, reject) => {
    // -ss 5 seeks to 5 seconds, -vframes 1 outputs one frame
    const cmd = `ffmpeg -y -ss 5 -i "${videoPath}" -vframes 1 -vf "scale=320:-1" "${thumbnailPath}"`;
    exec(cmd, (error, stdout, stderr) => {
      if (error) {
        console.error('Error generating thumbnail:', error, stderr);
        reject(error);
      } else {
        resolve();
      }
    });
  });
} 


================================================
File: video-watcher/src/App.jsx
================================================
import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { getStatus, getVideos, getMemoryState, queryMemory, searchVideos } from './api';

function App() {
  const navigate = useNavigate();
  const [status, setStatus] = useState(null);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [memoryState, setMemoryState] = useState(null);
  const [activeTab, setActiveTab] = useState('videos');
  const [memoryQuery, setMemoryQuery] = useState('');
  const [memoryResponse, setMemoryResponse] = useState('');
  const [queryLoading, setQueryLoading] = useState(false);
  
  // Search state
  const [searchQuery, setSearchQuery] = useState('');
  const [searchResults, setSearchResults] = useState([]);
  const [searchLoading, setSearchLoading] = useState(false);
  const [searchError, setSearchError] = useState(null);

  const [currentPage, setCurrentPage] = useState(1);
  const [sortOrder, setSortOrder] = useState('desc'); // 'desc' for most recent, 'asc' for oldest
  const VIDEOS_PER_PAGE = 12;

  // Fetch server status and videos on component mount
  useEffect(() => {
    async function fetchData() {
      try {
        setLoading(true);
        const statusData = await getStatus();
        setStatus(statusData);
        
        const videosData = await getVideos();
        setVideos(videosData);
        
        // Fetch memory state
        if (activeTab === 'memory') {
          const memoryData = await getMemoryState();
          setMemoryState(memoryData);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error fetching data:', err);
        setError('Failed to load data. Please check if the server is running.');
      } finally {
        setLoading(false);
      }
    }

    fetchData();
    
    // Poll for updates every 10 seconds
    const intervalId = setInterval(fetchData, 10000);
    
    return () => clearInterval(intervalId);
  }, [activeTab]);

  // Handle memory query submission
  const handleMemoryQuery = async (e) => {
    e.preventDefault();
    
    if (!memoryQuery.trim()) return;
    
    try {
      setQueryLoading(true);
      const result = await queryMemory(memoryQuery);
      setMemoryResponse(result.response);
    } catch (err) {
      console.error('Error querying memory:', err);
      setMemoryResponse('Sorry, there was an error processing your query.');
    } finally {
      setQueryLoading(false);
    }
  };

  // Handle search form submission
  const handleSearchSubmit = async (e) => {
    e.preventDefault();
    if (!searchQuery.trim()) return;
    
    setSearchLoading(true);
    setSearchResults([]);
    setSearchError(null);
    
    try {
      const data = await searchVideos(searchQuery);
      setSearchResults(data.results || []);
    } catch (err) {
      console.error('Search failed:', err);
      setSearchError(err.message || 'An unknown error occurred during search.');
    } finally {
      setSearchLoading(false);
    }
  };

  // Handle navigation to discussion page
  const handleNavigateToDiscussion = (filename) => {
    navigate(`/discuss/${filename}?query=${encodeURIComponent(searchQuery)}`);
  };

  // Handle direct navigation to discussion from video card
  const handleDirectVideoDiscussion = (filename, videoFileName) => {
    // Create a default query based on the video name
    const defaultQuery = `Tell me about this video: ${videoFileName}`;
    navigate(`/discuss/${filename}?query=${encodeURIComponent(defaultQuery)}`);
  };

  // Render a video card
  const VideoCard = ({ video }) => {
    const { id, videoFileName, processedAt, analysis } = video;
    // Construct the video URL (same as used in VideoDiscussion)
    const videoUrl = `/videos/${encodeURIComponent(videoFileName)}`;
    // Use the generated thumbnail as the poster
    const thumbName = videoFileName.replace(/\.[^/.]+$/, '.jpg');
    const posterUrl = `/thumbnails/${encodeURIComponent(thumbName)}`;
    const fallbackPoster = "/video-placeholder.jpg";

    return (
      <div className="card">
        <div className="video-thumbnail" style={{ width: '100%', aspectRatio: '16/9', background: '#000', borderRadius: '4px', overflow: 'hidden', marginBottom: '1rem' }}>
          <video
            src={videoUrl}
            poster={posterUrl}
            style={{ width: '100%', height: '100%', objectFit: 'cover', display: 'block' }}
            controls={false}
            tabIndex={-1}
            preload="metadata"
            onError={e => { e.target.poster = fallbackPoster; }}
          />
        </div>
        <h3>{videoFileName}</h3>
        <p><strong>Processed:</strong> {new Date(processedAt).toLocaleString()}</p>
        
        {analysis && (
          <div>
            <h4>Analysis</h4>
            {analysis.summary && (
              <p><strong>Summary:</strong> {analysis.summary}</p>
            )}
            
            {analysis.topics && analysis.topics.length > 0 && (
              <div>
                <strong>Topics:</strong>
                <ul>
                  {analysis.topics.map((topic, index) => (
                    <li key={index}>{topic}</li>
                  ))}
                </ul>
              </div>
            )}
            
            {analysis.tags && analysis.tags.length > 0 && (
              <div>
                <strong>Tags:</strong>{' '}
                {analysis.tags.map(tag => (
                  <span key={tag} style={{ 
                    display: 'inline-block',
                    margin: '0 4px 4px 0',
                    padding: '2px 8px',
                    backgroundColor: '#444',
                    borderRadius: '12px',
                    fontSize: '0.8rem'
                  }}>
                    {tag}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
        
        <button 
          className="continue-discussion-btn"
          onClick={() => handleDirectVideoDiscussion(videoFileName.replace(/\.[^/.]+$/, '.json'), videoFileName)}
        >
          Chat with Video
        </button>
      </div>
    );
  };

  // Render a search result card
  const SearchResultCard = ({ result }) => {
    return (
      <div className="card result-card">
        <h3>{result.videoFileName}</h3>
        <p><strong>Processed:</strong> {new Date(result.processedAt).toLocaleString()}</p>
        <p><strong>Relevance Score:</strong> {result.score.toFixed(2)}</p>
        <p><strong>Justification:</strong> {result.justification}</p>
        <button 
          className="continue-discussion-btn"
          onClick={() => handleNavigateToDiscussion(result.filename)}
        >
          Continue Discussion
        </button>
      </div>
    );
  };

  // Sort and paginate videos
  const sortedVideos = [...videos].sort((a, b) => {
    if (sortOrder === 'desc') {
      return new Date(b.processedAt) - new Date(a.processedAt);
    } else {
      return new Date(a.processedAt) - new Date(b.processedAt);
    }
  });
  const totalPages = Math.ceil(sortedVideos.length / VIDEOS_PER_PAGE);
  const paginatedVideos = sortedVideos.slice((currentPage - 1) * VIDEOS_PER_PAGE, currentPage * VIDEOS_PER_PAGE);

  if (loading && !status) {
    return <div className="container">Loading...</div>;
  }

  if (error) {
    return (
      <div className="container">
        <h1>Video Watcher</h1>
        <div className="card" style={{ backgroundColor: '#442222' }}>
          <h3>Error</h3>
          <p>{error}</p>
        </div>
      </div>
    );
  }

  return (
    <div className="container">
      <h1>Video Watcher</h1>
      
      {status && (
        <div className="card">
          <h2>Server Status</h2>
          <p>
            Status: <span className={`status status-active`}>{status.status}</span>
          </p>
          <p><strong>Watching folder:</strong> {status.watchFolder}</p>
          <p><strong>Dataset folder:</strong> {status.datasetFolder}</p>
        </div>
      )}
      
      {/* Search card - place it before the tabs */}
      <div className="card search-card">
        <h3>Search Video Journals</h3>
        <form onSubmit={handleSearchSubmit} className="search-form">
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Ask about your past recordings..."
            className="search-input"
            disabled={searchLoading}
          />
          <button
            type="submit"
            disabled={searchLoading || !searchQuery.trim()}
            className="search-submit"
          >
            {searchLoading ? 'Searching...' : 'Search'}
          </button>
        </form>
        {searchError && <p className="error-text">{searchError}</p>}
      </div>
      
      {/* Display search results if available */}
      {searchResults.length > 0 && (
        <div className="search-results">
          <h2>Search Results ({searchResults.length})</h2>
          <div className="video-list">
            {searchResults.map((result, index) => (
              <SearchResultCard key={index} result={result} />
            ))}
          </div>
        </div>
      )}
      
      {/* Show "no results" message if search was performed but returned nothing */}
      {!searchLoading && searchResults.length === 0 && searchQuery && searchError === null && (
        <div className="card"><p>No relevant videos found for your query.</p></div>
      )}
      
      <div className="tabs">
        <button 
          className={activeTab === 'videos' ? 'active' : ''} 
          onClick={() => setActiveTab('videos')}
        >
          Videos
        </button>
        <button 
          className={activeTab === 'memory' ? 'active' : ''} 
          onClick={() => setActiveTab('memory')}
        >
          Memory Portal
        </button>
      </div>
      
      {/* Only show regular video list if no search results are displayed */}
      {activeTab === 'videos' && searchResults.length === 0 && (
        <div>
          <h2>Processed Videos ({videos.length})</h2>
          <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
            <label htmlFor="sortOrder" style={{ marginRight: 8 }}>Sort by:</label>
            <select id="sortOrder" value={sortOrder} onChange={e => { setSortOrder(e.target.value); setCurrentPage(1); }}>
              <option value="desc">Most Recent</option>
              <option value="asc">Oldest First</option>
            </select>
          </div>
          {videos.length === 0 ? (
            <div className="card">
              <p>No videos have been processed yet. Record a video in OBS and save it to the watched folder.</p>
            </div>
          ) : (
            <>
              <div className="video-list">
                {paginatedVideos.map(video => (
                  <VideoCard key={video.id} video={video} />
                ))}
              </div>
              <div style={{ display: 'flex', justifyContent: 'center', marginTop: 16 }}>
                <button onClick={() => setCurrentPage(p => Math.max(1, p - 1))} disabled={currentPage === 1}>Previous</button>
                <span style={{ margin: '0 12px' }}>Page {currentPage} of {totalPages}</span>
                <button onClick={() => setCurrentPage(p => Math.min(totalPages, p + 1))} disabled={currentPage === totalPages}>Next</button>
              </div>
            </>
          )}
        </div>
      )}
      
      {activeTab === 'memory' && (
        <div className="memory-portal">
          <h2>Memory Portal</h2>
          
          <div className="card">
            <h3>Query Memory System</h3>
            <p>Ask questions about the memory system or the insights it has gathered.</p>
            
            <form onSubmit={handleMemoryQuery} className="memory-form">
              <input
                type="text"
                value={memoryQuery}
                onChange={(e) => setMemoryQuery(e.target.value)}
                placeholder="Ask about what the system remembers..."
                className="memory-input"
              />
              <button 
                type="submit" 
                disabled={queryLoading}
                className="memory-submit"
              >
                {queryLoading ? 'Thinking...' : 'Ask'}
              </button>
            </form>
            
            {memoryResponse && (
              <div className="memory-response">
                <h4>Response:</h4>
                <p>{memoryResponse}</p>
              </div>
            )}
          </div>
          
          <div className="memory-state">
            <h3>Memory State</h3>
            
            {memoryState ? (
              <div className="memory-cards">
                <div className="card">
                  <h4>Working Memory</h4>
                  <details>
                    <summary>Established Facts ({memoryState.workingMemory?.established_facts?.length || 0})</summary>
                    <ul>
                      {memoryState.workingMemory?.established_facts?.map((fact, idx) => (
                        <li key={idx}>{fact}</li>
                      )) || <li>No established facts yet</li>}
                    </ul>
                  </details>
                  <details>
                    <summary>Corroborated Hypotheses ({memoryState.workingMemory?.corroborated_hypotheses?.length || 0})</summary>
                    <ul>
                      {memoryState.workingMemory?.corroborated_hypotheses?.map((hyp, idx) => (
                        <li key={idx}>{hyp}</li>
                      )) || <li>No corroborated hypotheses yet</li>}
                    </ul>
                  </details>
                  <details>
                    <summary>Untested Hypotheses ({memoryState.workingMemory?.untested_hypotheses?.length || 0})</summary>
                    <ul>
                      {memoryState.workingMemory?.untested_hypotheses?.map((hyp, idx) => (
                        <li key={idx}>{hyp}</li>
                      )) || <li>No untested hypotheses yet</li>}
                    </ul>
                  </details>
                </div>
                
                <div className="card">
                  <h4>Long-Term Memory</h4>
                  {memoryState.longTermMemory?.profile_summary && (
                    <div>
                      <strong>Profile Summary:</strong>
                      <p>{memoryState.longTermMemory.profile_summary}</p>
                    </div>
                  )}
                  <details>
                    <summary>Skills & Knowledge</summary>
                    <pre className="memory-json">
                      {JSON.stringify(memoryState.longTermMemory?.skills_and_knowledge || {}, null, 2)}
                    </pre>
                  </details>
                  <details>
                    <summary>Preferences & Habits</summary>
                    <pre className="memory-json">
                      {JSON.stringify(memoryState.longTermMemory?.preferences_and_habits || {}, null, 2)}
                    </pre>
                  </details>
                </div>
              </div>
            ) : (
              <div className="card">
                <p>Loading memory state...</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

export default App; 


================================================
File: video-watcher/src/api.js
================================================
/**
 * API client for the Video Watcher application
 */

/**
 * Get the server status
 * @returns {Promise<Object>} Status object
 */
export async function getStatus() {
  const response = await fetch('/api/status');
  if (!response.ok) {
    throw new Error('Failed to fetch status');
  }
  return response.json();
}

/**
 * Get all processed videos
 * @returns {Promise<Array>} Array of video objects
 */
export async function getVideos() {
  const response = await fetch('/api/videos');
  if (!response.ok) {
    throw new Error('Failed to fetch videos');
  }
  const data = await response.json();
  return data.videos || [];
}

/**
 * Get the current memory state
 * @returns {Promise<Object>} Memory state object
 */
export async function getMemoryState() {
  const response = await fetch('/api/memory');
  if (!response.ok) {
    throw new Error('Failed to fetch memory state');
  }
  return response.json();
}

/**
 * Query the memory system conversationally
 * @param {string} query - The user's query about the memory system
 * @returns {Promise<Object>} Response from the memory system
 */
export async function queryMemory(query) {
  const response = await fetch('/api/memory/query', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });
  
  if (!response.ok) {
    throw new Error('Failed to query memory system');
  }
  
  return response.json();
}

/**
 * Search through video analyses based on a natural language query
 * @param {string} query - The search query
 * @returns {Promise<Array>} Array of relevant video objects
 */
export async function searchVideos(query) {
  const response = await fetch('/api/search', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query }),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Search request failed' }));
    throw new Error(errorData.error || 'Failed to perform search');
  }
  
  return response.json(); // Should return { results: [...] }
}

/**
 * Continue discussion with a specific video, including memory context
 * @param {string} query - The original search query
 * @param {string} filename - The filename of the video to continue discussion with
 * @returns {Promise<Object>} Video and memory context for continuation
 */
export async function continueDiscussion(query, filename) {
  const response = await fetch('/api/videos/continue-discussion', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ query, filename }),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Continue discussion request failed' }));
    throw new Error(errorData.error || 'Failed to continue discussion with video');
  }
  
  return response.json();
}

/**
 * Send a chat message in the video discussion
 * @param {string} message - The user's message
 * @param {Array} history - Previous messages in the conversation
 * @param {Object} videoContext - Context about the video
 * @param {Object} memoryContext - Memory state context
 * @returns {Promise<Object>} Response from the AI
 */
export async function sendChatMessage(message, history, videoContext, memoryContext) {
  const response = await fetch('/api/videos/chat', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      message,
      history,
      videoContext,
      memoryContext
    }),
  });
  
  if (!response.ok) {
    const errorData = await response.json().catch(() => ({ error: 'Chat request failed' }));
    throw new Error(errorData.error || 'Failed to process chat message');
  }
  
  return response.json();
} 


================================================
File: video-watcher/src/main.jsx
================================================
import React from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import App from './App.jsx';
import VideoDiscussion from './components/VideoDiscussion.jsx';
import './styles/index.css';

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />} />
        <Route path="/discuss/:filename" element={<VideoDiscussion />} />
      </Routes>
    </BrowserRouter>
  </React.StrictMode>
); 


================================================
File: video-watcher/src/components/VideoDiscussion.jsx
================================================
import React, { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate, useLocation } from 'react-router-dom';
import { continueDiscussion, sendChatMessage } from '../api';

function VideoDiscussion() {
  const { filename } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const query = new URLSearchParams(location.search).get('query') || '';
  
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [discussionData, setDiscussionData] = useState(null);
  const [messages, setMessages] = useState([]);
  const [currentMessage, setCurrentMessage] = useState('');
  const [isProcessing, setIsProcessing] = useState(false);
  
  // Ref for auto-scrolling messages
  const messagesEndRef = useRef(null);
  
  // Function to scroll to bottom of messages
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  
  // Auto-scroll when messages update
  useEffect(() => {
    scrollToBottom();
  }, [messages]);
  
  useEffect(() => {
    const loadDiscussion = async () => {
      if (!filename || !query) {
        setError('Missing required parameters');
        setLoading(false);
        return;
      }
      
      try {
        setLoading(true);
        const data = await continueDiscussion(query, filename);
        setDiscussionData(data);
        
        // Initialize with system message and original query
        setMessages([
          {
            type: 'system',
            content: `Discussion about: "${query}" with video "${data.videoContext.videoFileName}"`
          },
          {
            type: 'user',
            content: query
          }
        ]);
        
        // Make initial assistant response using the chat API
        setIsProcessing(true);
        try {
          const initialResponse = await sendChatMessage(
            query,
            [{ type: 'user', content: query }],
            data.videoContext,
            data.memoryContext
          );
          
          setMessages(prev => [
            ...prev,
            {
              type: 'assistant',
              content: initialResponse.response
            }
          ]);
        } catch (chatError) {
          console.error('Error getting initial response:', chatError);
          setMessages(prev => [
            ...prev,
            {
              type: 'assistant',
              content: 'I apologize, but I encountered an error while preparing my response. Please try again or ask a different question.'
            }
          ]);
        } finally {
          setIsProcessing(false);
        }
        
        setError(null);
      } catch (err) {
        console.error('Error loading discussion:', err);
        setError(err.message || 'Failed to load discussion data');
      } finally {
        setLoading(false);
      }
    };
    
    loadDiscussion();
  }, [filename, query]);
  
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    if (!currentMessage.trim() || isProcessing) return;
    
    // Add user message
    const updatedMessages = [
      ...messages,
      { type: 'user', content: currentMessage }
    ];
    
    setMessages(updatedMessages);
    setCurrentMessage('');
    setIsProcessing(true);
    
    try {
      // Send message to the API for processing
      // Filter out system messages and only send user/assistant messages
      const chatHistory = updatedMessages.filter(msg => msg.type === 'user' || msg.type === 'assistant');
      
      const response = await sendChatMessage(
        currentMessage,
        chatHistory,
        discussionData.videoContext,
        discussionData.memoryContext
      );
      
      // Add assistant response
      setMessages(prev => [
        ...prev,
        { 
          type: 'assistant', 
          content: response.response
        }
      ]);
    } catch (error) {
      console.error('Error sending chat message:', error);
      
      // Add error message
      setMessages(prev => [
        ...prev,
        { 
          type: 'assistant', 
          content: 'I apologize, but I encountered an error while processing your message. Please try again.'
        }
      ]);
    } finally {
      setIsProcessing(false);
    }
  };
  
  const getVideoUrl = () => {
    if (!discussionData || !discussionData.videoContext.videoFileName) return null;
    
    // In a real implementation, you would have a proper URL to the video file
    // This is a placeholder that assumes videos are served from a /videos endpoint
    return `/videos/${encodeURIComponent(discussionData.videoContext.videoFileName)}`;
  };
  
  const goBack = () => {
    navigate('/');
  };
  
  if (loading) {
    return (
      <div className="container">
        <h1>Loading Discussion...</h1>
        <div className="loading-spinner"></div>
      </div>
    );
  }
  
  if (error) {
    return (
      <div className="container">
        <h1>Error Loading Discussion</h1>
        <div className="card" style={{ backgroundColor: '#442222' }}>
          <p>{error}</p>
          <button onClick={goBack}>Return to Home</button>
        </div>
      </div>
    );
  }
  
  return (
    <div className="container discussion-page">
      <header className="discussion-header">
        <button className="back-button" onClick={goBack}>&larr; Back to Search</button>
        <h1>Video Discussion</h1>
      </header>
      
      <div className="discussion-layout">
        <section className="video-section">
          <div className="video-container">
            <h2>{discussionData.videoContext.videoFileName}</h2>
            <p><strong>Processed:</strong> {new Date(discussionData.videoContext.processedAt).toLocaleString()}</p>
            
            {/* Video element - in a real implementation, you would have proper video URLs */}
            <div className="video-player">
              {getVideoUrl() ? (
                <video 
                  controls 
                  width="100%" 
                  src={getVideoUrl()}
                  poster="/video-placeholder.jpg"
                >
                  Your browser does not support the video tag.
                </video>
              ) : (
                <div className="video-placeholder">
                  <p>Video preview not available</p>
                  <p className="small">The actual video file can be found at: {discussionData.videoContext.videoFileName}</p>
                </div>
              )}
            </div>
            
            <div className="video-info">
              <h3>Video Information</h3>
              {discussionData.videoContext.summary && (
                <div className="info-section">
                  <h4>Summary</h4>
                  <p>{discussionData.videoContext.summary}</p>
                </div>
              )}
              
              {discussionData.videoContext.topics && discussionData.videoContext.topics.length > 0 && (
                <div className="info-section">
                  <h4>Topics</h4>
                  <ul>
                    {discussionData.videoContext.topics.map((topic, idx) => (
                      <li key={idx}>{topic}</li>
                    ))}
                  </ul>
                </div>
              )}
              
              {discussionData.videoContext.insights && discussionData.videoContext.insights.length > 0 && (
                <div className="info-section">
                  <h4>Insights</h4>
                  <ul>
                    {discussionData.videoContext.insights.map((insight, idx) => (
                      <li key={idx}>
                        {insight.insight}
                        {insight.basis && <span className="basis">Based on: {insight.basis}</span>}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        </section>
        
        <section className="chat-section">
          <div className="messages-container">
            {messages.map((msg, index) => (
              <div key={index} className={`message ${msg.type}`}>
                <div className="message-content">{msg.content}</div>
              </div>
            ))}
            {isProcessing && (
              <div className="message assistant">
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          
          <form onSubmit={handleSubmit} className="discussion-form">
            <input
              type="text"
              value={currentMessage}
              onChange={(e) => setCurrentMessage(e.target.value)}
              placeholder="Continue the discussion..."
              className="discussion-input"
              disabled={isProcessing}
            />
            <button 
              type="submit" 
              className="discussion-submit"
              disabled={isProcessing || !currentMessage.trim()}
            >
              Send
            </button>
          </form>
        </section>
      </div>
    </div>
  );
}

export default VideoDiscussion; 






```

