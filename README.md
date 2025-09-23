# AI Guild Hackathon - PokeWar
### Solution by team TanVi - Tanish Chudiwal, Rajavi Gujarathi and Nishitha Dharmabotla

### Task:- 
Develop an AI-powered targeting system that combines computer vision
and natural language processing to:
1. Parse tactical orders from headquarters containing mission-specific
directives
2. Analyze battlefield imagery to identify Pokemon species and locations
3. Generate precise targeting coordinates based on mission parameters
4. Minimize collateral damage and ammunition waste

### Working demo with gradio for you to try out:-
    https://legendarray-pokehacke.hf.space/
(running on cpu)

### Generated data and model weights:-
    https://drive.google.com/drive/folders/1l3qMh0HDB6FHJWYbTNDZLRh9HXN4_Mvg?usp=drive_link
(only available for 8 more months, till May 2026)

### Approach:- 
1. Step 1 - generate data. in the problem statement, some examples of the complex orders from HQ have been given. 
so our first thought was to generate synthetic data. 

Generating image data (`image_gen.py` and `augment_composites.py`) :- 
   1. collected the images of the 4 types of pokemon (Ref images folder)
   2. removed their backgrounds
   3. collected 5k different background images (from kaggle) {we have used test set images from here for high quality and variety:- https://www.kaggle.com/datasets/nguyenquocdungk16hl/bg-20o }
   4. wrote a python script which will do the following tasks for each background - 
      1. pick random pokemon and a random reference image of that pokemon
      2. resize the reference image
      3. place it randomly on the image
      4. place total 0 - 8 pokemon cutouts per image, randomly and without overlap
      5. Note down all their positions in a neat coco formatted json file
   5. Following that we performed some image data augmentation for increasing robustness:-
      1. adding random circles with 15% opacity of colours orange, green, purple and yellow in an attempt to confuse the model
      2. adding random salt and pepper noise
      3. adding random gaussian noise
      4. adding straight and squiggly lines
   
Generating text data (`prompt_gen.py`):-
   1. Once the images were generated, we knew about the existence of which pokemon is available in which image. 
   2. After handcrafting a few examples and extensive prompt engineering.
   3. We used a free Grok4 api call and generated 5012 high quality orders which are similar to the creative, tricky and out of the box HQ orders given in the pdf
   4. Once this data was generated, slightly cleaned and 1% of it was manually verified to be good, we used it for training.

2. Step 2 - deciding models to be used
   1. after an elaborate discussion and a series of experiments
   2. experiments with YOLOv11, ORB, SAM, we settled onto YOLOv12 (59M) for the object detection task
   3. and experiments with TFIDF + xgboost, TFIDF + MLP, Sentence embedding + MLP, Longformer, Distilled big bird, BigBird with LGBM, and finally we settled on Google's BigBird model base Roberta(400M)
   4. together our final models have 459M parameters, below the 500M limit

3. Step 3 - training the YOLOv12
   1. **Data Preparation** – Converted COCO-format annotations to YOLO format, normalized bounding boxes, and ensured zero-based class IDs.  
   2. **Dataset Split** – Used an 80/20 train-validation split with a fixed random seed for reproducibility.  
   3. **Training Setup** – Initialized from a pretrained checkpoint and trained using Ultralytics’ `YOLO.train(...)` interface with validation monitoring.  
   4. **Model Selection** – Chose the higher-capacity `yolo12x` model for better accuracy on the 15k-image dataset, accepting slower training and inference.  


4. Step 4 - training the BigBird
   1. **Data Setup:** Loads JSON dataset, maps Pokémon classes to numeric labels, splits into train/val/test (70/15/15) with stratified sampling 
   2. **Model Configuration:** Fine-tunes google/bigbird-roberta-base for 4-class classification with 1024 max token length and efficient tokenization
   3. **Training:** Uses minimal TrainingArguments (3 epochs, 2e-5 LR, batch size 4) to avoid version compatibility issues, no in-training evaluation to enable robust functioning on wider range of libraries and more devices
   4. **Evaluation:** Post-training assessment on all splits with accuracy metrics, classification reports, and confusion matrices
   5. **Output:** Saves trained model, tokenizer, and evaluation results to specified directory for competition submission

5. Step 5 - integrating it all together
   1. **YOLOv12** was trained to detect all the pokemon in the image regardless of the target and return them in a clean csv format
   2. **BigBird** was trained to predict the target from the given sentence and output the respective targets in clean json format
   3. `final_processing.py` contains the simple code which reads the outputs from the respective json and csv generated above, marks the centers of the target and saves it into a csv file
   4. `check_sub.py` plots all the centers generated onto the images for manual assessment before final submission

### Code Quality
- **Modular Codebase:** All major functionality is separated into individual Python modules, following a logical, maintainable architecture.
- **Documentation:** Every function and script includes clear docstrings and inline comments explaining intent and logic.

### Repository Structure
- `Data Generation scripts`
  - `image_gen.py`: Synthetic image creation pipeline
  - `augment_composites.py`: Image Data augmentation scripts
  - `prompt_gen.py`: Order text generation logic
- `final_processing.py`: Output post-processing and coordinate extraction
- `complete_pipeline.py`: complete end to end pipeline which takes the test images and prompts as input and gives center containing csv as output
- `check_submissions.py`: Visualization/QA script

### Running Instructions to be completed
1. Clone the repository and install requirements from requirements.txt
2. Install the test images and prompts from the kaggle competition: https://www.kaggle.com/competitions/the-poke-war-hackathon-ai-guild-recuritment-hack
3. Install the model weights from drive link above (only available for 8 more months, till May 2026)
4. ensure you set correct paths in the `complete_pipeline.py`
5. Run complete_pipeline.py

