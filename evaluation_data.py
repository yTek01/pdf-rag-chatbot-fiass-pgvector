evaluation_dataset = [
    {
        "question": "What does the Langmuir adsorption isotherm indicate about the inhibitor?",
        "answer": "It indicates that the inhibitor follows a Langmuir adsorption behavior on the stainless steel surface."
    },
    {
        "question": "How does the corrosion rate change with increasing inhibitor concentration?",
        "answer": "The corrosion rate decreases significantly as inhibitor concentration increases."
    },
    {
        "question": "What experimental method was used to examine the surface morphology?",
        "answer": "The surface morphology was examined using Scanning Electron Microscopy (SEM)."
    },
    {
        "question": "What information is provided in Table 3 regarding thermodynamic parameters?",
        "answer": "Table 3 shows values for ΔG, ΔH, and ΔS indicating spontaneous and exothermic adsorption behavior."
    },
    {
        "question": "Which figure presents the polarization curve and what does it reveal?",
        "answer": "Figure 2 presents the polarization curve showing decreased current density with inhibitor presence."
    },
    {
        "question": "Describe the role of FTIR analysis in the study.",
        "answer": "FTIR analysis was used to identify functional groups involved in the adsorption of the inhibitor on metal surfaces."
    },
    {
        "question": "What is the main conclusion derived from the thermodynamic analysis?",
        "answer": "The thermodynamic analysis concludes that the adsorption is spontaneous and predominantly physical in nature."
    },
    {
        "question": "Based on the SEM image, what is the difference between the inhibited and uninhibited surface?",
        "answer": "The uninhibited surface shows visible corrosion pits, while the inhibited surface appears smoother and less damaged."
    },
    {
        "question": "What was the optimal inhibitor concentration observed in the experiments?",
        "answer": "The optimal inhibitor concentration was observed at 100 ppm, where the corrosion rate was minimized."
    },
    {
        "question": "How does temperature affect the adsorption efficiency according to the results?",
        "answer": "Higher temperatures reduce adsorption efficiency, suggesting physical adsorption dominates the mechanism."
    }
]
evaluation_dataset_hbs = [
    {
        "question": "What nutrients make red meat a valuable part of a balanced diet?",
        "answer": "Red meat is rich in high biological value protein, vitamin B12, iron, zinc, niacin, vitamin B6, and phosphorus."
    },
    {
        "question": "How does the fat content of trimmed lean red meat compare to untrimmed cuts?",
        "answer": "Trimmed lean red meat has significantly lower fat content, generally under 7%, compared to untrimmed cuts that may contain up to 37% fat."
    },
    {
        "question": "Which bioactive compounds in red meat have antioxidant properties?",
        "answer": "Red meat contains carnosine, anserine, ubiquinone (coenzyme Q10), and glutathione, all of which exhibit antioxidant properties."
    },
    {
        "question": "What impact does animal feeding practice have on the omega-3 content of red meat?",
        "answer": "Pasture-fed animals produce meat with higher omega-3 fatty acid content compared to grain-fed animals."
    },
    {
        "question": "Which organ meat is the richest source of vitamin A and folate?",
        "answer": "Liver is the richest source of vitamin A and folate among organ meats."
    },
    {
        "question": "How does red meat contribute to iron absorption?",
        "answer": "Red meat provides well-absorbed haem-iron and enhances non-haem iron absorption due to the 'meat factor'."
    },
    {
        "question": "What distinguishes mutton from other red meats in terms of nutrient density?",
        "answer": "Mutton is particularly nutrient-dense, containing higher levels of thiamin, vitamins B6 and B12, phosphorus, iron, and copper compared to other red meats."
    },
    {
        "question": "Why is red meat a good source of carnitine?",
        "answer": "Carnitine is found abundantly in skeletal muscle, with lamb containing up to 209 mg/100g and beef around 60 mg/100g."
    },
    {
        "question": "How does the protein quality of red meat compare to plant proteins?",
        "answer": "Red meat has a higher protein digestibility and complete amino acid profile, with a PDCAAS close to 0.9, whereas most plant proteins score between 0.5 and 0.7."
    },
    {
        "question": "Which vitamin is notably low in red meat but high in liver?",
        "answer": "Vitamin A is low in lean meat tissue but very high in liver, making it an important source when consuming organ meats."
    }
]

evaluation_dataset_layout = [
    {
        "question": "What is the primary purpose of LayoutParser?",
        "answer": "LayoutParser is an open-source library designed to simplify the use of deep learning models for document image analysis (DIA), including layout detection, OCR, and document processing."
    },
    {
        "question": "What challenges in document image analysis does LayoutParser aim to address?",
        "answer": "LayoutParser addresses challenges such as lack of reusable infrastructure, domain-specific customization difficulties, and fragmented, undocumented processing pipelines."
    },
    {
        "question": "Which deep learning models are commonly used in LayoutParser for layout detection?",
        "answer": "LayoutParser uses models like Faster R-CNN and Mask R-CNN built on the Detectron2 framework for layout detection tasks."
    },
    {
        "question": "How does LayoutParser facilitate OCR integration?",
        "answer": "LayoutParser provides a unified API to plug-and-play OCR tools like Tesseract and Google Cloud Vision, and it also includes a custom CNN-RNN OCR model."
    },
    {
        "question": "What are the three core layout data structures used in LayoutParser?",
        "answer": "The three core data structures are Coordinate, TextBlock, and Layout, which together enable flexible and hierarchical representation of document layouts."
    },
    {
        "question": "What datasets are used to train the models available in LayoutParser’s Model Zoo?",
        "answer": "Datasets include PubLayNet, PRImA, Newspaper Navigator, TableBank, and HJDataset, each tailored to different document types and domains."
    },
    {
        "question": "How does LayoutParser support model customization and training?",
        "answer": "It supports efficient annotation using object-level active learning and allows both fine-tuning of pre-trained models and training from scratch."
    },
    {
        "question": "Describe one example use case of LayoutParser presented in the paper.",
        "answer": "LayoutParser was used to digitize historical Japanese financial documents by training custom models to detect columns and tokens, and using OCR reorganization to improve recognition accuracy."
    },
    {
        "question": "What tools does LayoutParser provide for visualizing layout and OCR results?",
        "answer": "LayoutParser includes APIs for overlaying layout boxes and OCR text on images and for recreating structured versions of documents for review and debugging."
    },
    {
        "question": "What is the goal of the LayoutParser community platform?",
        "answer": "The community platform aims to promote reusability by allowing users to share pre-trained models and full document digitization pipelines."
    }
]


evaluation_dataset_edu = [
    {
        "question": "What key problem in education does RAG aim to solve?",
        "answer": "RAG addresses the issue of hallucinations in LLM-based chatbots by integrating information retrieval to enhance factual consistency."
    },
    {
        "question": "What is the main limitation of standard LLMs in educational settings?",
        "answer": "Standard LLMs struggle with knowledge updates and are prone to hallucinations, which compromises reliability in education."
    },
    {
        "question": "Which data sources were used to identify relevant publications for the study?",
        "answer": "The study used Scopus, Web of Science, and Google Scholar to identify relevant publications on RAG chatbots in education."
    },
    {
        "question": "How many unique studies were included in the final dataset after screening?",
        "answer": "A total of 47 research papers were qualified for analysis after removing duplicates and unrelated works."
    },
    {
        "question": "Which thematic domain had the highest number of RAG chatbot applications?",
        "answer": "The 'Access to source knowledge' domain had the highest number of RAG chatbot applications, with 20 publications."
    },
    {
        "question": "What were the most commonly used LLMs for educational RAG chatbots?",
        "answer": "OpenAI's GPT models were the most frequently used, followed by Meta's LLaMA models."
    },
    {
        "question": "What aspect of evaluation was most frequently addressed in the studies?",
        "answer": "Evaluation criteria such as faithfulness, relevance, user acceptance, and accuracy were most frequently addressed."
    },
    {
        "question": "Which evaluation framework was adopted in multiple studies?",
        "answer": "RAGAS was the most commonly adopted evaluation framework, assessing faithfulness, answer relevance, and context relevance."
    },
    {
        "question": "What recommendation was made for evaluating chatbots that support learning?",
        "answer": "The authors recommend comparing learning outcomes through competency tests between users and non-users of the chatbot."
    },
    {
        "question": "What is a noted weakness in current evaluations of RAG chatbots for education?",
        "answer": "A major weakness is the lack of evaluation on whether the chatbot achieves its intended educational outcome."
    }
]


