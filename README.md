# Flask Image Classifier on Vercel

This is a Flask application that uses TensorFlow to classify sports images. The application is configured to be deployed on Vercel's serverless platform.

## Features

- Image classification using EfficientNetB3 model
- Integration with Google Drive for image storage
- Supabase for prediction history storage
- Responsive UI with drag-and-drop image upload

## Local Development

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up environment variables in `.env` file
4. Run the application:
   ```
   flask run
   ```

## Deploying to Vercel

1. Install Vercel CLI:
   ```
   npm install -g vercel
   ```

2. Login to Vercel:
   ```
   vercel login
   ```

3. Deploy the application:
   ```
   vercel --prod
   ```

## Environment Variables

The following environment variables need to be set in Vercel:

- `GOOGLE_CREDENTIALS`: Service account credentials for Google Drive API
- `DRIVE_FOLDER_ID`: Google Drive folder ID for storing uploaded images
- `SUPABASE_URL`: Supabase project URL
- `SUPABASE_KEY`: Supabase API key
- `MODEL_PATH`: Path to the TensorFlow model file

## Project Structure

- `app.py`: Main Flask application
- `model/`: Contains the TensorFlow model
- `static/`: Static assets (CSS, JavaScript)
- `templates/`: HTML templates
- `vercel.json`: Vercel deployment configuration

## Notes on Serverless Deployment

- The application uses `/tmp` directory for temporary file storage on Vercel
- TensorFlow model is loaded at cold start, which may cause initial delay
- Static files are served directly by Vercel