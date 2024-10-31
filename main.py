"""
Main.py    
"""

# Import statements
import os

# LangChain Imports necessary for RAG
from langchain_openai import OpenAIEmbeddings # handle word embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
import langchain_core.prompts.chat

# Chain Extractors:
from langchain.retrievers.document_compressors.chain_extract import LLMChainExtractor
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever

# New imports
from langchain import hub

# New imports for creating document chains + retrieval
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# Runnable PassThrough - node connections with no ops
from langchain_core.runnables import RunnablePassthrough

# Combine or stuffing chain
from langchain.chains.combine_documents import create_stuff_documents_chain

# Load environment variables
from dotenv import load_dotenv
load_dotenv()




# --------- Example ---------

# One-Shot example:

# Previous example care of plan:
example_care_plan = """
Patient Summary:
The patient is over 85 years old, female, and uses a mobility aid regularly. She does not report any health problems that limit her activities or require her to stay at home, and she does not need regular help from others. However, she does not have a reliable person to count on when help is needed. Her Gait and TUG test results indicate high risk, with gait speeds of 6.349 m/s and 5.634 m/s, and TUG test times of 10 and 30 seconds. Her standing pose is reported as not stable.

Frailty Status:
The patient's frailty status is mixed. She maintains some independence and does not report health problems that limit her activities. However, her age, use of a mobility aid, high-risk gait speed, prolonged TUG test times, and unstable standing pose all indicate potential frailty. The key areas of concern are her mobility and balance, which increase her risk of falls and injuries.

Care Recommendations:

1. Physical Therapy:
   - Rationale: The patient's mobility and balance issues, as indicated by her use of a mobility aid, high-risk gait speed, prolonged TUG test times, and unstable standing pose, suggest a need for physical therapy.
   - Implementation: Arrange for a physical therapist to assess the patient's needs and develop a personalized exercise program. This could include strength training, balance exercises, and gait training.
   - Challenges: The patient may resist physical therapy due to fear of falls or discomfort. Encourage her to participate by explaining the benefits and ensuring the exercises are safe and appropriate for her abilities.

2. Home Safety Assessment:
   - Rationale: The patient's mobility and balance issues increase her risk of falls at home.
   - Implementation: Arrange for a home safety assessment by a professional. They can recommend modifications such as installing grab bars, removing tripping hazards, and improving lighting.
   - Challenges: The patient may resist changes to her home. Explain the benefits and involve her in the decision-making process to increase her acceptance.

3. Social Support:
   - Rationale: The patient does not have a reliable person to count on when help is needed, which could pose a risk in emergencies.
   - Implementation: Explore options for social support, such as community programs, volunteer services, or professional caregiving services. Consider a personal emergency response system for added safety.
   - Challenges: The patient may resist accepting help from others. Encourage her to accept support by explaining the benefits and ensuring her privacy and independence are respected.

4. Mobility Aid Adjustment:
   - Rationale: The patient's use of a mobility aid and her high-risk gait speed suggest that her current mobility aid may not be optimal.
   - Implementation: Arrange for a professional to assess the patient's mobility aid and make necessary adjustments or recommend a different aid.
   - Challenges: The patient may resist changes to her mobility aid. Explain the benefits and involve her in the decision-making process to increase her acceptance.

Safety Considerations:
Ensure the patient's home is safe for her mobility and balance issues. This includes removing tripping hazards, improving lighting, and installing safety features such as grab bars and non-slip mats. Encourage the patient to use her mobility aid at all times and to wear non-slip shoes.

Monitoring and Evaluation:
Regularly monitor the patient's mobility, balance, and overall health. This could include regular physical therapy assessments, home safety checks, and health check-ups. Adjust the care plan as needed based on these assessments.

Resources and Support Services:
Consider resources such as community programs, volunteer services, professional caregiving services, and personal emergency response systems. These can provide social support, help with daily activities, and emergency assistance.

Additional Assessments:
Further assessments may be needed to fully understand the patient's frailty status. This could include cognitive function, nutrition, and social support assessments. Consult with healthcare professionals as needed.

This care plan is based on the information provided and is intended to guide caregivers in supporting the patient. It is not a substitute for professional medical advice. Always consult with healthcare professionals for a comprehensive assessment and personalized care plan.
"""

# --------- Function to Generate Care Plan ---------

def generate_frailty_care_plan(
    first_gait_test_speed: float,
    first_gait_test_time: float,
    first_tug_test_time: float,
    gait_speed_test_risk: str,
    second_gait_test_speed: float,
    second_gait_test_time: float,
    second_tug_test_time: float,
    tug_test_risk: str,
    older_than_85: bool,
    is_male: bool,
    has_limiting_health_problems: bool,
    needs_regular_help: bool,
    has_homebound_health_problems: bool,
    has_close_help: bool,
    uses_mobility_aid: bool
):
    # Initialize OpenAI Embeddings
    embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

    # Connect to PineCone vector store
    vectorstore = PineconeVectorStore(
        index_name=os.environ["INDEX_NAME"],
        embedding=embeddings
    )

    # Create the retriever
    retriever = vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 10})

    # Create the chat model
    chat = ChatOpenAI(verbose=True, temperature=0, model="gpt-4")

    # Create the prompts
    first_invocation_prompt = PromptTemplate.from_template("""
    You are an expert chatbot focused on frailty care, analyzing a patient's condition based on their PRISMA-7 survey responses and test results. Your task is to provide a factual analysis based solely on the given information. Do not make assumptions or infer information that is not explicitly stated.

    Patient's PRISMA-7 Responses and GAIT/TUG Test Results:
    <input>
    {input}
    </input>

    Relevant context from the knowledge base:
    <context>
    {context}
    </context>

    Please provide a detailed analysis considering only the information provided above. Address the following points:

    1. Frailty status: For each response from Patient's PRISMA-7 Responses and GAIT/TUG Test Results, think about how it contributes to the patient's frailty status.
    2. Overall frailty assessment: Summarize what the given responses and test results indicate about the patient's frailty status.
    3. Key areas of concern: Identify the most critical aspects that need addressing, based solely on the provided information.
    4. Potential risks: Discuss potential risks that are directly related to the information given.
    5. Care needs: Suggest interventions or support strategies that are relevant to the specific issues mentioned in the input.
    6. Interrelations: Explain how the different aspects of the patient's condition, as described in the input, may impact each other.

    In your analysis:
    - Be specific and refer only to the information provided in the input and context.
    - If the input or context doesn't provide sufficient information for any point, clearly state this lack of information.
    - Do not make assumptions or infer details that are not explicitly stated.
    - If you use information from the context, cite the source.

    Your goal is to provide an accurate understanding of the patient's frailty status based strictly on the given information. If there are gaps in the information or if more assessment is needed, state this clearly.

    Remember, do not provide any medical advice. Your role is to analyze the given information to support the development of a care plan by healthcare professionals.
    """)

    second_invocation_prompt = PromptTemplate.from_template("""
    You are an expert chatbot focused on frailty care, tasked with creating a comprehensive, personalized care plan. Your goal is to synthesize the provided analysis into an actionable, tailored care plan that supports both the caretaker and the frailty patient.

    You avoid humor or casual language due to the seriousness of the topic.

    You are provided the following information and analysis of the patient's condition.
    Patient's PRISMA-7 Responses, and Gait and TUG Test results:
    <input>
    {input}
    </input>

    I have conducted the following analysis of the patient's condition:
    <analysis>
    {analysis}
    </analysis>

    Based on this analysis, create a comprehensive care plan that addresses the specific needs and circumstances of this frailty patient. 

    First, you must begin your care plan by summarizing all the responses from the PRISMA-7 survey, and the Gait and TUG test results.
    Next, continue by saying "As a caretaker, you should consider the following:".
    Then, the care plan should:

    1. Provide a concise summary of the patient's overall frailty status, highlighting key areas of concern.

    2. Outline 4-5 key care recommendations. For each recommendation:
       a) Clearly state the recommendation
       b) Explain the rationale behind it, citing specific aspects of the patient's condition
       c) Provide detailed, practical steps for implementation
       d) Identify potential challenges and suggest strategies to overcome them

    3. Address safety considerations specific to this patient's situation, including both home safety and broader health and wellbeing measures.

    4. Suggest a monitoring and evaluation plan to track the patient's progress and adjust care as needed.

    5. Recommend specific resources or support services that would be particularly beneficial for this patient.

    6. Identify any areas where additional assessment or professional consultation might be necessary, explaining why.

    Throughout your care plan:
    - Ensure each recommendation is clearly linked to specific aspects of the patient's condition.
    - Prioritize interventions that address the most critical aspects of the patient's frailty status.
    - Consider the interplay between physical, cognitive, and social aspects of the patient's health.
    - Include both short-term interventions for immediate concerns and long-term strategies for ongoing care.
    - Provide clear, actionable guidance that can be readily implemented by caregivers.

    Your care plan should be comprehensive, practical, and tailored to both the patient's needs and the caretaker's ability to implement it.

    If there are any uncertainties or gaps in your knowledge, please say so and do not make up information. Clearly state what additional information or next steps would be required from healthcare providers.

    Your care plan should be comprehensive yet practical, providing clear guidance that can be readily implemented by caregivers while also serving as a valuable resource for healthcare professionals involved in the patient's care.

    Remember, your plan should be tailored to the patient's needs, and also meaningful to help caretakers as well.

    While knowledgeable about frailty care, you stay within your role of developing a care plan to support the caretaker and frailty patient, without providing definitive medical advice. Should there be any uncertainty, you should state this, and suggest the user to speak with a licensed healthcare professional.
    
    Here is an example format of a care plan:
    <example>
    {example}
    </example>
    
    """)

    # Create the chains
    stuff_documents_chain = create_stuff_documents_chain(chat, first_invocation_prompt)
    qa = create_retrieval_chain(retriever=retriever, combine_docs_chain=stuff_documents_chain)

    # Prepare the input data
    input_data = {
        "Are you older than 85 years?": "Yes" if older_than_85 else "No",
        "Are you male?": "Yes" if is_male else "No",
        "In general, do you have any health problems that require you to limit your activities?": "Yes" if has_limiting_health_problems else "No",
        "Do you need someone to help you on a regular basis?": "Yes" if needs_regular_help else "No",
        "In general, do you have any health problems that require you to stay at home?": "Yes" if has_homebound_health_problems else "No",
        "If you need help, can you count on someone close to you?": "Yes" if has_close_help else "No",
        "Do you regularly use a stick, walker or wheelchair to move about?": "Yes" if uses_mobility_aid else "No",
        "First Gait Test speed": f"{first_gait_test_speed} meters per second (m/s).",
        "First Gait Test time": f"{first_gait_test_time} seconds",
        "First TUG Test time": f"{first_tug_test_time} seconds",
        "Gait Speed Test Risk": gait_speed_test_risk,
        "Second Gait Test speed": f"{second_gait_test_speed} meters per second (m/s).",
        "Second Gait Test time": f"{second_gait_test_time} seconds",
        "Second TUG Test time": f"{second_tug_test_time} seconds",
        "TUG Test Risk": tug_test_risk,
    }

    # Run the first invocation
    first_result = qa.invoke(input={"input": str(input_data)})

    # Run the second invocation
    final_care_plan = chat.invoke(second_invocation_prompt.format(
        input=str(input_data),
        analysis=first_result["answer"],
        example=example_care_plan
    ))

    # Format the care plan as a big string
    care_plan_string = f"""
Care Plan:
{final_care_plan.content}

Sources used:
{chr(10).join(f"{i+1}. {source}" for i, source in enumerate(sorted(set(doc.metadata["source"] for doc in first_result["context"]))))}
"""

    return care_plan_string

# Example usage:
if __name__ == "__main__":
    result = generate_frailty_care_plan(
        first_gait_test_speed=6.349205195858935,
        first_gait_test_time=230,
        first_tug_test_time=300,
        gait_speed_test_risk="High",
        second_gait_test_speed=5.633802514207732,
        second_gait_test_time=200,
        second_tug_test_time=300,
        tug_test_risk="High",
        older_than_85=False,
        is_male=True,
        has_limiting_health_problems=True,
        needs_regular_help=False,
        has_homebound_health_problems=False,
        has_close_help=False,
        uses_mobility_aid=True
    )

    print(result)  # This will print the entire care plan string
