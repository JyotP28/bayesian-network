
import streamlit as st
import pandas as pd
import json
import os
import altair as alt
import google.generativeai as genai
from engine import VetBayesianEngine

st.set_page_config(page_title="Vet Diagnostic Simulator", layout="wide")
st.title("Veterinary Bayesian Diagnostic Simulator")

try:
    api_key = st.secrets["GEMINI_API_KEY"]
    genai.configure(api_key=api_key)
except KeyError:
    st.error("Missing API Key. Please add GEMINI_API_KEY to your .streamlit/secrets.toml file.")
    st.stop()

# loads the data and the engine
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

# initializes session state
if "active_signalment" not in st.session_state:
    st.session_state.active_signalment = None
if "active_findings" not in st.session_state:
    st.session_state.active_findings = None
if "active_results" not in st.session_state:
    st.session_state.active_results = None

# gemini instructions
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

    generation_config = {"response_mime_type": "application/json", "temperature": 0.0}
    model = genai.GenerativeModel('gemini-3.1-flash-lite', generation_config=generation_config)
    
    try:
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Failed to parse with Gemini: {e}")
        st.stop()

# render the UI
def render_results(results):
    st.subheader("Ranked Differentials & Clinical Reasoning")
    for diff in results:
        match_pct = float(diff['posterior_probability'].replace('%', ''))
        
        with st.expander(f"{diff['name']} -- Match: {diff['posterior_probability']} (Pre-test Baseline: {diff['pre_test_probability']})"):
            
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("### Supporting Evidence")
                if diff["drivers"]:
                    st.markdown("\n".join(diff["drivers"]))
                else:
                    st.write("*No supporting findings observed.*")
                
                st.markdown("---")
                st.markdown("### Recommended Next Steps")
                if diff["next_steps"]:
                    for step in diff["next_steps"]:
                        st.markdown(f"* {step}")
                else:
                    st.write("*No specific confirmatory tests listed.*")
                    
            with c2:
                st.markdown("### Missing Classic Signs")
                st.caption("Hallmark signs of this disease that are ABSENT in this patient.")
                if diff["missing_hallmarks"]:
                    st.markdown("\n".join(diff["missing_hallmarks"]))
                else:
                    st.write("*Patient has all expected hallmark signs.*")

                st.markdown("---")
                st.markdown("### Uncharacteristic Findings")
                st.caption("Active findings in this patient that DO NOT fit this disease profile.")
                if diff["penalties"]:
                    st.markdown("\n".join(diff["penalties"]))
                else:
                    st.write("*No conflicting signs present.*")
            
            st.progress(int(match_pct) if match_pct <= 100 else 100)
            
            st.markdown("---")
            with st.expander("View Mathematical Trace"):
                trace = diff["math_trace"]
                
                st.markdown("#### 1. Adjusted Prior Probability")
                st.caption("Base prevalence multiplied by signalment risk factors.")
                
                if isinstance(trace['base_prevalence'], str):
                    st.latex(rf"\text{{Prior}} = 1.0 \times {trace['age_mult']} \text{{ (Age)}} \times {trace['sex_mult']} \text{{ (Sex)}} \times {trace['breed_mult']} \text{{ (Breed)}} = {trace['adjusted_prior']:.6f}")
                else:
                    st.latex(rf"\text{{Prior}} = {trace['base_prevalence']} \times {trace['age_mult']} \text{{ (Age)}} \times {trace['sex_mult']} \text{{ (Sex)}} \times {trace['breed_mult']} \text{{ (Breed)}} = {trace['adjusted_prior']:.6f}")
                
                st.markdown("#### 2. Likelihood (Log Sum Method)")
                st.caption("Probabilities are converted to natural logs and summed to prevent floating-point underflow, then exponentiated back.")
                
                for f in trace["findings"]:
                    st.write(f"* **{f['name']}** ({f['tier']}): $P = {f['prob']} \\rightarrow \\ln(P) = {f['log_prob']:.4f}$")
                
                st.write("") 
                st.latex(rf"\sum \ln(P) = {trace['log_likelihood']:.4f} \implies e^{{{trace['log_likelihood']:.4f}}} = {trace['likelihood']:.4e}")
                
                st.markdown("#### 3. Raw Score & Posterior Normalization")
                st.caption("The Raw Score is normalized against the sum of all Raw Scores in the system to calculate the final % match.")
                
                st.latex(rf"\text{{Raw Score}} = \text{{Prior}} \times \text{{Likelihood}} = {trace['raw_score']:.4e}")
                st.latex(rf"\text{{Posterior}} = \frac{{\text{{Raw Score}}}}{{\text{{Sum of All Raw Scores}} ({trace['total_weight']:.4e})}} = {match_pct / 100:.4f} \approx {diff['posterior_probability']}")

# UI Tabs
tab1, tab2, tab3 = st.tabs(["Case Parser", "Bayesian Trajectory", "Database Browser"])

with tab1:
    st.markdown("Paste raw clinical notes, lab results, or textbook cases here. Using Gemini-Flash-3.1, your raw case input will be formatted into JSON to allow our program to utilize the necessary information")
    case_text = st.text_area("Raw Case Input", height=200, placeholder="CASE 1\nSignalment: 10 yr old, CM, Miniature poodle\nHistory: Presented for teeth cleaning...\nAbnormalities: WBC 18.1, ALP 578...")
    
    if st.button("Parse & Run Diagnostics", type="primary", key="parse_btn"):
        with st.spinner("Translating clinical text..."):
            parsed_data = extract_case_with_ai(case_text, all_breeds, all_findings)
            
        st.success("Case successfully translated into structured data.")
        with st.expander("View extracted JSON data"):
            st.json(parsed_data)
        
        st.session_state.active_signalment = parsed_data["signalment"]
        st.session_state.active_findings = parsed_data["observed_findings"]
        st.session_state.active_results = engine.calculate_differentials(
            st.session_state.active_signalment, 
            st.session_state.active_findings
        )
        
        render_results(st.session_state.active_results)

with tab2:
    st.header("Clinical Reasoning Trajectory")
    st.markdown("Watch the Bayesian network update its confidence sequentially as each clinical sign is evaluated. This mimics the cognitive shift a clinician experiences as new data arrives.")
    
    if st.session_state.active_findings is None:
        st.info("Run a case in the Parser tab to view the diagnostic trajectory.")
    else:
        trajectory_data = engine.get_trajectory_data(
            st.session_state.active_signalment, 
            st.session_state.active_findings
        )
        
        df = pd.DataFrame(trajectory_data)
        df_melted = df.melt(id_vars=["Step"], var_name="Disease", value_name="Probability")
        
        chart = alt.Chart(df_melted).mark_line(point=True).encode(
            x=alt.X('Step:N', sort=None, title='Sequential Clinical Findings'), 
            y=alt.Y('Probability:Q', title='Posterior Probability (%)'),
            color=alt.Color('Disease:N', legend=alt.Legend(title=None, orient="bottom")),
            tooltip=['Step', 'Disease', alt.Tooltip('Probability:Q', format='.2f')]
        ).properties(
            height=500
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)

with tab3:
    st.header("Disease Database Browser")
    st.markdown("Audit the underlying clinical data, prior probabilities, and symptom weighting tiers powering the Bayesian engine.")
    
    disease_dict = {d["disease_metadata"]["official_name"]: d for d in data.get("diseases", []) if "disease_metadata" in d}
    selected_db_disease = st.selectbox("Select a Disease Record to View:", options=list(disease_dict.keys()))
    
    if selected_db_disease:
        db_data = disease_dict[selected_db_disease]
        meta = db_data.get("disease_metadata", {})
        
        st.subheader(meta.get("official_name", "Unknown"))
        st.caption(f"Citation: {meta.get('textbook_source_chapter_or_citation', 'N/A')}")
        
        c1, c2, c3 = st.columns(3)
        with c1:
            prevalence = meta.get('estimated_population_base_prevalence')
            prev_display = f"{prevalence * 100:.2f}%" if isinstance(prevalence, (int, float)) else "Missing"
            st.metric("Base Prevalence (Prior)", prev_display)
        with c2:
            st.metric("Primary Species", meta.get('primary_species', 'N/A').title())
        with c3:
            st.metric("Internal ID", meta.get('disease_id', 'N/A'))

        st.divider()

        st.markdown("### Signalment Risk Multipliers")
        st.caption("Baseline risk is 1.0. Values > 1.0 indicate predisposition; values < 1.0 indicate reduced risk.")
        
        sig = db_data.get("signalment_risk_multipliers", {})
        
        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown("**Age Brackets**")
            for age, mult in sig.get("age_brackets", {}).items():
                st.write(f"* {age.replace('_', ' ').title()}: `{mult}x`")
        with sc2:
            st.markdown("**Biological Sex**")
            for sex, mult in sig.get("biological_sex_and_status", {}).items():
                st.write(f"* {sex.replace('_', ' ').title()}: `{mult}x`")
        with sc3:
            st.markdown("**Predisposed Breeds**")
            breeds = sig.get("predisposed_breeds", {})
            clean_breeds = {k: v for k, v in breeds.items() if k != "instructional_note"}
            for breed, mult in clean_breeds.items():
                if breed != "default_all_other_breeds":
                    st.write(f"* {breed.replace('_', ' ').title()}: `{mult}x`")

        st.divider()

        st.markdown("### Clinical Findings & Lab Abnormalities")
        st.caption("Categorized by textbook frequency. These tiers determine the Bayesian likelihood penalties/rewards.")
        
        symptoms = db_data.get("clinical_history_and_physical_exam_symptoms", {})
        labs = db_data.get("routine_laboratory_abnormalities_cbc_chem_ua", {})
        
        tc1, tc2, tc3 = st.columns(3)
        
        def render_tier(col_title, tier_key, dict1, dict2):
            items = []
            if isinstance(dict1.get(tier_key), list): items.extend(dict1[tier_key])
            if isinstance(dict2.get(tier_key), list): items.extend(dict2[tier_key])
            
            st.markdown(f"**{col_title}**")
            if items:
                for item in items:
                    st.write(f"* {item.replace('_', ' ').title()}")
            else:
                st.write("*None listed*")

        with tc1:
            render_tier("Classic (60-100%)", "common_tier_60_to_100_percent_occurrence", symptoms, labs)
        with tc2:
            render_tier("Expected (20-59%)", "less_common_tier_20_to_59_percent_occurrence", symptoms, labs)
        with tc3:
            render_tier("Possible (1-19%)", "uncommon_or_rare_tier_1_to_19_percent_occurrence", symptoms, labs)

        st.divider()

        st.markdown("### Specific Confirmatory Diagnostics")
        tests = db_data.get("specific_confirmatory_diagnostic_tests", [])
        
        if tests:
            for test in tests:
                with st.expander(f"Test: {test.get('test_name', 'Unnamed Test')}"):
                    st.write(f"**Sensitivity:** {test.get('test_sensitivity_decimal', 'N/A')} | **Specificity:** {test.get('test_specificity_decimal', 'N/A')}")
                    st.markdown("**Diagnostic Criteria:**")
                    st.write(test.get('textbook_diagnostic_criteria_summary', 'No criteria provided.'))
        else:
            st.info("No specific confirmatory tests listed for this disease.")