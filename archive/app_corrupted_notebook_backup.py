{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "86d6482a-9d0a-417e-84f6-954e89bbd6e9",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "Warning: You are sending unauthenticated requests to the HF Hub. Please set a HF_TOKEN to enable higher rate limits and faster downloads.\n"
     ]
    },
    {
     "data": {
      "application/vnd.jupyter.widget-view+json": {
       "model_id": "06e1b632a82446ecbae94d4f25ab498a",
       "version_major": 2,
       "version_minor": 0
      },
      "text/plain": [
       "Loading weights:   0%|          | 0/103 [00:00<?, ?it/s]"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "Question:\n",
      "Does Anand have GCP experience?\n",
      "\n",
      "Answer:\n",
      "Has Anand worked with cloud platforms like GCP?\n",
      "Yes. Anand implemented a Customer Churn Prediction solution using Google Cloud Platform, Google Cloud Storage, Google Dataproc, and PySpark. The solution processed datasets up to 1 million customer records in a distributed cloud environment.\n",
      "\n",
      "Which projects prove Anand's analytics skills?\n",
      "Key analytics projects include:\n",
      "\t•\tCustomer Churn Prediction using PySpark & GCP\n",
      "\t•\tRetail Customer Behavior Analysis & High-Value Customer Prediction\n",
      "\t•\tRetail Sales Performance Analysis\n",
      "\t•\tBrazil Retail Analytics Capstone\n",
      "\t•\tNorthwind Sales Analytics & Visualization\n",
      "These projects demonstrate business analytics, machine learning, predictive modeling, customer analytics, and business intelligence capabilities.\n",
      "\n",
      "Supporting Evidence:\n",
      "\n",
      "--- Evidence 1 ---\n",
      "Has Anand worked with cloud platforms like GCP?\n",
      "Yes. Anand implemented a Customer Churn Prediction solution using Google Cloud Platform, Google Cloud Storage, Google Dataproc, and PySpark. The solution processed datasets up to 1 million customer records in a distributed cloud environment.\n",
      "\n",
      "Which projects prove Anand's analytics skills?\n",
      "Key analytics projects include:\n",
      "\t•\tCustomer Churn Prediction using PySpark & GCP\n",
      "\t•\tRetail Customer Behavior Analysis & High-Value Customer Prediction\n",
      "\t•\tRetail Sales Performance Analysis\n",
      "\t•\tBrazil Retail Analytics Capstone\n",
      "\t•\tNorthwind Sales Analytics & Visuali\n",
      "Source: KnowledgeBase/06_interviews:/Q&A.md\n",
      "\n",
      "--- Evidence 2 ---\n",
      "Can Anand work independently with minimum supervision?\n",
      "Yes. Throughout his career, Anand has managed dealer networks, customer service initiatives, business transformation projects, and operational programs requiring independent decision-making, stakeholder management, and execution.\n",
      "\n",
      "What measurable value can Anand bring in the first 90 days?\n",
      "Within the first 90 days, Anand can understand business operations, identify key performance drivers, improve reporting and dashboards, analyze customer and sales data, identify improvement opportunities, and recommend data-driven actions that improve op\n",
      "Source: KnowledgeBase/06_interviews:/Q&A.md\n",
      "\n",
      "--- Evidence 3 ---\n",
      "Does Anand have enough cloud and machine learning experience?\n",
      "Anand has hands-on project experience with machine learning and cloud platforms. While he is still growing his expertise in these areas, he has already demonstrated the ability to build end-to-end analytics solutions using cloud infrastructure and machine learning techniques.\n",
      "Source: KnowledgeBase/06_interviews:/Q&A.md\n",
      "\n"
     ]
    }
   ],
   "source": [
    "import chromadb\n",
    "from sentence_transformers import SentenceTransformer\n",
    "\n",
    "VECTOR_DB_PATH = \"vector_db\"\n",
    "COLLECTION_NAME = \"ask_anand\"\n",
    "\n",
    "embedding_model = SentenceTransformer(\"all-MiniLM-L6-v2\")\n",
    "client = chromadb.PersistentClient(path=VECTOR_DB_PATH)\n",
    "collection = client.get_collection(name=COLLECTION_NAME)\n",
    "\n",
    "\n",
    "def retrieve_context(query, n_results=5):\n",
    "    query_embedding = embedding_model.encode(query).tolist()\n",
    "\n",
    "    results = collection.query(\n",
    "        query_embeddings=[query_embedding],\n",
    "        n_results=n_results\n",
    "    )\n",
    "\n",
    "    return results\n",
    "\n",
    "\n",
    "def answer_question(query, n_results=5):\n",
    "    results = retrieve_context(query, n_results)\n",
    "\n",
    "    documents = results[\"documents\"][0]\n",
    "    metadatas = results[\"metadatas\"][0]\n",
    "\n",
    "    answer = f\"\"\"\n",
    "Question:\n",
    "{query}\n",
    "\n",
    "Answer:\n",
    "{documents[0].strip()}\n",
    "\n",
    "Supporting Evidence:\n",
    "\"\"\"\n",
    "\n",
    "    for i, doc in enumerate(documents[:3], 1):\n",
    "        source = metadatas[i - 1].get(\"source\", \"Unknown source\")\n",
    "        answer += f\"\\n--- Evidence {i} ---\\n\"\n",
    "        answer += doc.strip()[:600] + \"\\n\"\n",
    "        answer += f\"Source: {source}\\n\"\n",
    "\n",
    "    return answer\n",
    "\n",
    "\n",
    "if __name__ == \"__main__\":\n",
    "    test_question = \"Does Anand have GCP experience?\"\n",
    "    print(answer_question(test_question))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "edeb164c-acb6-4703-8540-7bb3bbb8e6e2",
   "metadata": {},
   "outputs": [
    {
     "ename": "NameError",
     "evalue": "name 'null' is not defined",
     "output_type": "error",
     "traceback": [
      "\u001b[0;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[0;31mNameError\u001b[0m                                 Traceback (most recent call last)",
      "Cell \u001b[0;32mIn[3], line 1\u001b[0m\n\u001b[0;32m----> 1\u001b[0m \u001b[38;5;28;01mfrom\u001b[39;00m \u001b[38;5;21;01mrag_engine\u001b[39;00m \u001b[38;5;28;01mimport\u001b[39;00m answer_question\n\u001b[1;32m      3\u001b[0m \u001b[38;5;28mprint\u001b[39m(answer_question(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mTell me about CARE Direct 24x7\u001b[39m\u001b[38;5;124m\"\u001b[39m))\n\u001b[1;32m      4\u001b[0m \u001b[38;5;28mprint\u001b[39m(answer_question(\u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhy should we hire Anand?\u001b[39m\u001b[38;5;124m\"\u001b[39m))\n",
      "File \u001b[0;32m~/rag_engine.py:135\u001b[0m\n\u001b[1;32m      1\u001b[0m {\n\u001b[1;32m      2\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcells\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m      3\u001b[0m   {\n\u001b[1;32m      4\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcell_type\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcode\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m      5\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mexecution_count\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m2\u001b[39m,\n\u001b[1;32m      6\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mid\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdf060972-104d-4e7c-b998-b4ae9bf8c158\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m      7\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {},\n\u001b[1;32m      8\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutputs\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m      9\u001b[0m     {\n\u001b[1;32m     10\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     11\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mapplication/vnd.jupyter.widget-view+json\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m     12\u001b[0m        \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmodel_id\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mbcc7b204b1894a2abdfe39a952219622\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     13\u001b[0m        \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion_major\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m2\u001b[39m,\n\u001b[1;32m     14\u001b[0m        \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion_minor\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m0\u001b[39m\n\u001b[1;32m     15\u001b[0m       },\n\u001b[1;32m     16\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtext/plain\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m     17\u001b[0m        \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mLoading weights:   0\u001b[39m\u001b[38;5;124m%\u001b[39m\u001b[38;5;124m|          | 0/103 [00:00<?, ?it/s]\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     18\u001b[0m       ]\n\u001b[1;32m     19\u001b[0m      },\n\u001b[1;32m     20\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {},\n\u001b[1;32m     21\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutput_type\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdisplay_data\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     22\u001b[0m     },\n\u001b[1;32m     23\u001b[0m     {\n\u001b[1;32m     24\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstdout\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     25\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutput_type\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mstream\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     26\u001b[0m      \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtext\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m     27\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     28\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mQuestion:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     29\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDoes Anand have GCP experience?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     30\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     31\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAnswer:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     32\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mHas Anand worked with cloud platforms like GCP?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     33\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mYes. Anand implemented a Customer Churn Prediction solution using Google Cloud Platform, Google Cloud Storage, Google Dataproc, and PySpark. The solution processed datasets up to 1 million customer records in a distributed cloud environment.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     34\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     35\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhich projects prove Anand\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124ms analytics skills?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     36\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mKey analytics projects include:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     37\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mCustomer Churn Prediction using PySpark & GCP\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     38\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mRetail Customer Behavior Analysis & High-Value Customer Prediction\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     39\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mRetail Sales Performance Analysis\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     40\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mBrazil Retail Analytics Capstone\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     41\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mNorthwind Sales Analytics & Visualization\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     42\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mThese projects demonstrate business analytics, machine learning, predictive modeling, customer analytics, and business intelligence capabilities.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     43\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     44\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSupporting Evidence:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     45\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     46\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m--- Evidence 1 ---\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     47\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mHas Anand worked with cloud platforms like GCP?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     48\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mYes. Anand implemented a Customer Churn Prediction solution using Google Cloud Platform, Google Cloud Storage, Google Dataproc, and PySpark. The solution processed datasets up to 1 million customer records in a distributed cloud environment.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     49\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     50\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhich projects prove Anand\u001b[39m\u001b[38;5;124m'\u001b[39m\u001b[38;5;124ms analytics skills?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     51\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mKey analytics projects include:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     52\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mCustomer Churn Prediction using PySpark & GCP\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     53\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mRetail Customer Behavior Analysis & High-Value Customer Prediction\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     54\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mRetail Sales Performance Analysis\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     55\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mBrazil Retail Analytics Capstone\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     56\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124m•\u001b[39m\u001b[38;5;130;01m\\t\u001b[39;00m\u001b[38;5;124mNorthwind Sales Analytics & Visuali\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     57\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSource: KnowledgeBase/06_interviews:/Q&A.md\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     58\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     59\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m--- Evidence 2 ---\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     60\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCan Anand work independently with minimum supervision?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     61\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mYes. Throughout his career, Anand has managed dealer networks, customer service initiatives, business transformation projects, and operational programs requiring independent decision-making, stakeholder management, and execution.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     62\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     63\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWhat measurable value can Anand bring in the first 90 days?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     64\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mWithin the first 90 days, Anand can understand business operations, identify key performance drivers, improve reporting and dashboards, analyze customer and sales data, identify improvement opportunities, and recommend data-driven actions that improve op\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     65\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSource: KnowledgeBase/06_interviews:/Q&A.md\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     66\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     67\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m--- Evidence 3 ---\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     68\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mDoes Anand have enough cloud and machine learning experience?\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     69\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAnand has hands-on project experience with machine learning and cloud platforms. While he is still growing his expertise in these areas, he has already demonstrated the ability to build end-to-end analytics solutions using cloud infrastructure and machine learning techniques.\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     70\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSource: KnowledgeBase/06_interviews:/Q&A.md\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     71\u001b[0m       \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m     72\u001b[0m      ]\n\u001b[1;32m     73\u001b[0m     }\n\u001b[1;32m     74\u001b[0m    ],\n\u001b[1;32m     75\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msource\u001b[39m\u001b[38;5;124m\"\u001b[39m: [\n\u001b[1;32m     76\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mimport chromadb\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     77\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mfrom sentence_transformers import SentenceTransformer\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     78\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     79\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     80\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mVECTOR_DB_PATH = \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mvector_db\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     81\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mCOLLECTION_NAME = \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mask_anand\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     82\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     83\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     84\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124membedding_model = SentenceTransformer(\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mall-MiniLM-L6-v2\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     85\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     86\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mclient = chromadb.PersistentClient(path=VECTOR_DB_PATH)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     87\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcollection = client.get_collection(name=COLLECTION_NAME)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     88\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     89\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     90\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdef retrieve_context(query, n_results=5):\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     91\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    query_embedding = embedding_model.encode(query).tolist()\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     92\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     93\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    results = collection.query(\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     94\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        query_embeddings=[query_embedding],\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     95\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        n_results=n_results\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     96\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    )\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     97\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     98\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    return results\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m     99\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    100\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    101\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdef answer_question(query, n_results=5):\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    102\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    results = retrieve_context(query, n_results)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    103\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    104\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    documents = results[\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mdocuments\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m][0]\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    105\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    metadatas = results[\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mmetadatas\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m][0]\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    106\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    107\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    combined_context = \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m.join(documents)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    108\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    109\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    answer = f\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    110\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mQuestion:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    111\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;132;01m{query}\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    112\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    113\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mAnswer:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    114\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m{\u001b[39m\u001b[38;5;124mdocuments[0].strip()}\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    115\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    116\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mSupporting Evidence:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    117\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    118\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    119\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    for i, doc in enumerate(documents[:3], 1):\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    120\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        source = metadatas[i - 1].get(\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124msource\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m, \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mUnknown source\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m)\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    121\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        answer += f\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn--- Evidence \u001b[39m\u001b[38;5;132;01m{i}\u001b[39;00m\u001b[38;5;124m ---\u001b[39m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    122\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        answer += doc.strip()[:600] + \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    123\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m        answer += f\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mSource: \u001b[39m\u001b[38;5;132;01m{source}\u001b[39;00m\u001b[38;5;130;01m\\\\\u001b[39;00m\u001b[38;5;124mn\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    124\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    125\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    return answer\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    126\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    127\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    128\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mif __name__ == \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m__main__\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124m:\u001b[39m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    129\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    test_question = \u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;124mDoes Anand have GCP experience?\u001b[39m\u001b[38;5;130;01m\\\"\u001b[39;00m\u001b[38;5;130;01m\\n\u001b[39;00m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    130\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m    print(answer_question(test_question))\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    131\u001b[0m    ]\n\u001b[1;32m    132\u001b[0m   },\n\u001b[1;32m    133\u001b[0m   {\n\u001b[1;32m    134\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcell_type\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcode\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[0;32m--> 135\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mexecution_count\u001b[39m\u001b[38;5;124m\"\u001b[39m: null,\n\u001b[1;32m    136\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mid\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m0d488670-6609-448a-b11a-d32c8a8ee983\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    137\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {},\n\u001b[1;32m    138\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124moutputs\u001b[39m\u001b[38;5;124m\"\u001b[39m: [],\n\u001b[1;32m    139\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124msource\u001b[39m\u001b[38;5;124m\"\u001b[39m: []\n\u001b[1;32m    140\u001b[0m   }\n\u001b[1;32m    141\u001b[0m  ],\n\u001b[1;32m    142\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmetadata\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m    143\u001b[0m   \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mkernelspec\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m    144\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mdisplay_name\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mPython [conda env:base] *\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    145\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlanguage\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    146\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mconda-base-py\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    147\u001b[0m   },\n\u001b[1;32m    148\u001b[0m   \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mlanguage_info\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m    149\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mcodemirror_mode\u001b[39m\u001b[38;5;124m\"\u001b[39m: {\n\u001b[1;32m    150\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mipython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    151\u001b[0m     \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m3\u001b[39m\n\u001b[1;32m    152\u001b[0m    },\n\u001b[1;32m    153\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mfile_extension\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m.py\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    154\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mmimetype\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mtext/x-python\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    155\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mname\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    156\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbconvert_exporter\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpython\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    157\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mpygments_lexer\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mipython3\u001b[39m\u001b[38;5;124m\"\u001b[39m,\n\u001b[1;32m    158\u001b[0m    \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mversion\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124m3.12.7\u001b[39m\u001b[38;5;124m\"\u001b[39m\n\u001b[1;32m    159\u001b[0m   }\n\u001b[1;32m    160\u001b[0m  },\n\u001b[1;32m    161\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbformat\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m4\u001b[39m,\n\u001b[1;32m    162\u001b[0m  \u001b[38;5;124m\"\u001b[39m\u001b[38;5;124mnbformat_minor\u001b[39m\u001b[38;5;124m\"\u001b[39m: \u001b[38;5;241m5\u001b[39m\n\u001b[1;32m    163\u001b[0m }\n",
      "\u001b[0;31mNameError\u001b[0m: name 'null' is not defined"
     ]
    }
   ],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b248fa06-9e09-4dbc-aac5-ef795adc0a9d",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python [conda env:base] *",
   "language": "python",
   "name": "conda-base-py"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
