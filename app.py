import streamlit as st
import json
import os
import google.generativeai as genai
from engine import VetBayesianEngine

st.set_page_config(page_title="Vet Diagnostic Simulator", layout="wide")
st.title("🩺 Veterinary Bayesian Diagnostic Simulator")

# --- SIDEBAR: API KEY CONFIG ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")
if api_key:
    genai.configure(api_key=api_key)

# --- LOAD DATA & ENGINE ---
@st.cache_data
def load_data():
    with open('diseases.json', 'r') as f: return json.load(f)

if not os.path.exists('diseases.json'):
    st.error("Missing diseases.json file.")
    st.stop()

data = load_data()
engine = VetBayesianEngine('diseases.json')

all_findings, all_breeds = set(), set()
for d in data.get("diseases", []):
    if "disease_metadata" not in d: continue
    for b in d.get("signalment_risk_multipliers", {}).get("predisposed_breeds", {}).keys():
        if b != "default_all_other_breeds": all_breeds.add(b)
    for tier in d.get("clinical_history_and_physical_exam_symptoms", {}).values():
        if isinstance(tier, list): all_findings.update(tier)
    for tier in d.get("routine_laboratory_abnormalities_cbc_chem_ua", {}).values():
        if isinstance(tier, list): all_findings.update(tier)

all_findings = sorted(list(all_findings))
all_breeds = sorted(list(all_breeds))

# --- LIVE GEMINI API FUNCTION ---
def extract_case_with_ai(case_text, allowed_breeds, allowed_findings):
    if not api_key:
        st.error("Please enter your Gemini API key in the sidebar.")
        st.stop()

    prompt = f"""
    You are a veterinary clinical data extraction tool. Read the raw clinical case and translate it into a strict JSON format.

    ### MAPPING RULES:
    1. AGE: Map the age to one of these EXACT strings: "young_adult", "mature", "senior".
    2. SEX: Map the sex to one of these EXACT strings: "male_intact", "male_castrated", "female_intact", "female_spayed". (Note: CM = male_castrated, FS = female_spayed).
    3. BREED: Map the breed to the closest match in this list: {allowed_breeds}. If not on the list, use "default_all_other_breeds".
    4. FINDINGS: Read the history, physical exam, and lab data. Translate abnormal findings into EXACT strings from this list: {allowed_findings}. 
       - Interpret lab values strictly. If ALP is elevated above reference, output "increased_alkaline_phosphatase". If USG is low, output "urine_specific_gravity_less_than_1_020".
       - Do not invent any finding strings not on the list.

    RAW CASE TO TRANSLATE:
    {case_text}
    
    ### OUTPUT FORMAT:
    Output ONLY valid JSON.
    {{
      "signalment": {{
        "age": "",
        "sex": "",
        "breed": ""
      }},
      "observed_findings": []
    }}
    """

    generation_config = {"response_mime_type": "application/json"}
    model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config=generation_config)
    
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Failed to parse with Gemini: {e}")
        st.stop()

# --- UI LAYOUT ---
tab1, tab2 = st.tabs(["📝 AI Case Parser (Paste Text)", "🎛️ Manual Entry"])

def render_results(results):
    st.subheader("Ranked Differentials & Clinical Reasoning")
    for diff in results:
        match_pct = float(diff['posterior_probability'].replace('%', ''))
        color = "green" if match_pct > 75 else "orange" if match_pct > 20 else "red"
        
        with st.expander(f"📊 {diff['name']} — Match: {diff['posterior_probability']} (Pre-test Baseline: {diff['pre_test_probability']})"):
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### 👍 Supporting Evidence")
                if diff["drivers"]:
                    st.markdown("\n".join(diff["drivers"]))
                else:
                    st.write("*No supporting findings observed.*")
                
                st.markdown("---")
                st.markdown("### 🔬 Recommended Next Steps")
                if diff["next_steps"]:
                    for step in diff["next_steps"]:
                        st.markdown(f"• {step}")
                else:
                    st.write("*No specific confirmatory tests listed.*")
                    
            with c2:
                st.markdown("### ⚠️ Missing Classic Signs")
                st.caption("Hallmark signs of this disease that are ABSENT in this patient.")
                if diff["missing_hallmarks"]:
                    st.markdown("\n".join(diff["missing_hallmarks"]))
                else:
                    st.write("*Patient has all expected hallmark signs.*")

                st.markdown("---")
                st.markdown("### ❌ Uncharacteristic Findings")
                st.caption("Active findings in this patient that DO NOT fit this disease profile.")
                if diff["penalties"]:
                    st.markdown("\n".join(diff["penalties"]))
                else:
                    st.write("*No conflicting signs present.*")
            
            st.progress(int(match_pct) if match_pct <= 100 else 100)
            
            # --- UPDATED MATHEMATICAL TRACE UI ---
            st.markdown("---")
            with st.expander("🧮 View Mathematical Trace"):
                trace = diff["math_trace"]
                
                st.markdown("#### 1. Adjusted Prior Probability")
                st.caption("Base prevalence multiplied by signalment risk factors.")
                st.latex(rf"\text{{Prior}} = {trace['base_prevalence']} \times {trace['age_mult']} \text{{ (Age)}} \times {trace['sex_mult']} \text{{ (Sex)}} \times {trace['breed_mult']} \text{{ (Breed)}} = {trace['adjusted_prior']:.6f}")
                
                st.markdown("#### 2. Likelihood (Log Sum Method)")
                st.caption("Probabilities are converted to natural logs and summed to prevent floating-point underflow, then exponentiated back.")
                
                for f in trace["findings"]:
                    st.write(f"• **{f['name']}** ({f['tier']}): $P = {f['prob']} \\rightarrow \\ln(P) = {f['log_prob']:.4f}$")
                
                st.write("") 
                st.latex(rf"\sum \ln(P) = {trace['log_likelihood']:.4f} \implies e^{{{trace['log_likelihood']:.4f}}} = {trace['likelihood']:.4e}")
                
                st.markdown("#### 3. Raw Score & Posterior Normalization")
                st.caption("The Raw Score is normalized against the sum of all Raw Scores in the system to calculate the final % match.")
                
                st.latex(rf"\text{{Raw Score}} = \text{{Prior}} \times \text{{Likelihood}} = {trace['raw_score']:.4e}")
                st.latex(rf"\text{{Posterior}} = \frac{{\text{{Raw Score}}}}{{\text{{Sum of All Raw Scores}} ({trace['total_weight']:.4e})}} = {match_pct / 100:.4f} \approx {diff['posterior_probability']}")

with tab1:
    st.markdown("Paste raw clinical notes, lab results, or textbook cases here.")
    case_text = st.text_area("Raw Case Input", height=200, placeholder="CASE 1\nSignalment: 10 yr old, CM, Miniature poodle\nHistory: Presented for teeth cleaning...\nAbnormalities: WBC 18.1, ALP 578...")
    
    if st.button("Parse & Run Diagnostics", type="primary"):
        with st.spinner("Gemini 3.1 Flash-Lite is translating clinical text..."):
            parsed_data = extract_case_with_ai(case_text, all_breeds, all_findings)
            
        st.success("Case successfully translated into structured data!")
        with st.expander("View extracted JSON data"):
            st.json(parsed_data)
        
        results = engine.calculate_differentials(parsed_data["signalment"], parsed_data["observed_findings"])
        render_results(results)

with tab2:
    st.markdown("Use this tab to manually toggle specific symptoms and test the engine.")
    
    col1, col2 = st.columns([1, 2])
    with col1:
        age = st.selectbox("Age Bracket", ["young_adult", "mature", "senior"])
        sex = st.selectbox("Biological Sex", ["male_intact", "male_castrated", "female_intact", "female_spayed"])
        breed = st.selectbox("Breed", ["default_all_other_breeds"] + all_breeds)
        selected_findings = st.multiselect("Active Findings", all_findings)
        
    with col2:
        if st.button("Run Manual Inference", type="primary"):
            if not selected_findings:
                st.warning("Please select at least one finding.")
            else:
                signalment = {"age": age, "sex": sex, "breed": breed}
                results = engine.calculate_differentials(signalment, selected_findings)
                render_results(results)