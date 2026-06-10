from transformers import pipeline 
from langchain_huggingface import HuggingFacePipeline
from langchain_core.prompts import PromptTemplate
from transformers.utils.logging import set_verbosity_error

set_verbosity_error()

summarization_pipeline = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
summarizer = HuggingFacePipeline(pipeline=summarization_pipeline)

refinement_pipeline = pipeline("summarization", model="facebook/bart-large", device=0)
refiner = HuggingFacePipeline(pipeline=refinement_pipeline)

qa_pipeline = pipeline("question-answering", model="deepset/roberta-base-squad2", device=0)

summary_template = PromptTemplate.from_template("Summarize the following text in a {length} way:\n\n{text}")

summarization_chain = summary_template | summarizer | refiner

text_to_summarize ="""The University of the West Indies (The UWI) began its journey in 1948, welcoming 
just 33 medical students to its first faculty, Medicine, at Mona, Jamaica.  Back then, it was 
the University College of the West Indies (UCWI), affiliated with the University of 
London.  To support the clinical training of its students, a hospital was essential, and so, 
in 1949, the University College Hospital of the West Indies (UCHWI) was established.  By 
1967, it had evolved into The University Hospital of the West Indies (UHWI), which has 
grown to become the largest teaching hospital in the Caribbean, sitting right next to the 
Mona campus.  UHWI has been the practical arm of the Faculty of Medical Sciences, 
offering care while doubling as a teaching and research facility. With around 580 beds, 
intensive care units, multiple operating theatres, and specialised services like 
telemedicine through the Hugh Wynter Fertility Management Unit, UHWI plays a critical 
role in delivering medical education and healthcare services. 
The UWI didn’t stop at Mona. In 1967, the same year UHWI got its current name, 
it expanded clinical training to its Cave Hill Campus in Barbados.  That move sparked a 
longstanding partnership with the Queen Elizabeth Hospital (QEH), which still serves as 
the teaching hospital for that campus.  Initially, medical students studied their basic 
sciences in Jamaica before heading to Barbados for clinical training. Today, the full 
medical programme is offered at Cave Hill, with QEH at the heart of both undergraduate 
and postgraduate education.  Much like UHWI, QEH supports not just clinical training 
but also research and professional development, with many of its medical staff serving 
as lecturers or clinical instructors. These collaborations reflect The UWI’s broader 
commitment to improving healthcare and training professionals to meet the region’s 
needs."""

length = input("\nEnter the length (short/medium/long): ")

summary = summarization_chain.invoke({"text": text_to_summarize, "length": length})

print("\n🔹 **Generated Summary:**")
print(summary)

while True:
    question = input("\nAsk a question about the summary (or type 'exit' to stop):\n")
    if question.lower() == "exit":
        break

    qa_result = qa_pipeline(question=question, context=summary)

    print("\n🔹 **Answer:**")
    print(qa_result["answer"])