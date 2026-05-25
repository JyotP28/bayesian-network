import json
import math
import os

class VetBayesianEngine:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        # Softened probabilities to prevent mathematical flatlining
        self.tier_probs = {
            "common_tier_60_to_100_percent_occurrence": 0.80,
            "less_common_tier_20_to_59_percent_occurrence": 0.40,
            "uncommon_or_rare_tier_1_to_19_percent_occurrence": 0.15
        }
        # Increased background noise prevents 0% certainties for atypical presentations
        self.background_noise = 0.10

    def calculate_differentials(self, patient_signalment, active_findings):
        results = {}
        total_posterior_weight = 0.0

        for disease in self.data["diseases"]:
            if "disease_metadata" not in disease: continue
            meta = disease["disease_metadata"]
            d_id = meta["disease_id"]
            name = meta["official_name"]
            
            p_disease = meta["estimated_population_base_prevalence"]
            sig_mults = disease["signalment_risk_multipliers"]
            
            age_mult = sig_mults.get("age_brackets", {}).get(patient_signalment.get('age'), 1.0)
            sex_mult = sig_mults.get("biological_sex_and_status", {}).get(patient_signalment.get('sex'), 1.0)
            breed_mult = sig_mults.get("predisposed_breeds", {}).get(patient_signalment.get('breed'), 
                            sig_mults.get("predisposed_breeds", {}).get("default_all_other_breeds", 1.0))
            
            adjusted_prior = p_disease * age_mult * sex_mult * breed_mult

            # Initialize the math trace dictionary for this disease
            math_trace = {
                "base_prevalence": p_disease,
                "age_mult": age_mult,
                "sex_mult": sex_mult,
                "breed_mult": breed_mult,
                "adjusted_prior": adjusted_prior,
                "findings": [],
                "log_likelihood": 0.0,
                "likelihood": 0.0,
                "raw_score": 0.0
            }

            log_likelihood = 0.0
            
            symptoms = disease.get("clinical_history_and_physical_exam_symptoms", {})
            labs = disease.get("routine_laboratory_abnormalities_cbc_chem_ua", {})
            
            all_mapped_findings = {}
            for tier, items in symptoms.items():
                if isinstance(items, list):
                    for item in items: all_mapped_findings[item] = tier
            for tier, items in labs.items():
                if isinstance(items, list):
                    for item in items: all_mapped_findings[item] = tier

            drivers = []
            penalties = []
            
            for finding in active_findings:
                if finding in all_mapped_findings:
                    tier = all_mapped_findings[finding]
                    p_finding_given_d = self.tier_probs.get(tier, self.background_noise)
                    tier_display = "Classic" if "common_tier" in tier else "Expected" if "less_common" in tier else "Possible"
                    drivers.append(f"• **{finding.replace('_', ' ').title()}** ({tier_display})")
                    
                    # Log the math step for matches
                    math_trace["findings"].append({
                        "name": finding.replace('_', ' ').title(),
                        "tier": tier_display,
                        "prob": p_finding_given_d,
                        "log_prob": math.log(p_finding_given_d)
                    })
                else:
                    p_finding_given_d = self.background_noise
                    penalties.append(f"• **{finding.replace('_', ' ').title()}**")
                    
                    # Log the math step for penalties
                    math_trace["findings"].append({
                        "name": finding.replace('_', ' ').title(),
                        "tier": "Unmatched Penalty",
                        "prob": p_finding_given_d,
                        "log_prob": math.log(p_finding_given_d)
                    })
                
                log_likelihood += math.log(p_finding_given_d)

            missing_hallmarks = []
            for finding, tier in all_mapped_findings.items():
                if "common_tier" in tier and finding not in active_findings:
                    missing_hallmarks.append(f"• **{finding.replace('_', ' ').title()}**")

            next_steps = []
            for test in disease.get("specific_confirmatory_diagnostic_tests", []):
                test_name = test.get("test_name", "").replace('_', ' ').title()
                desc = test.get("textbook_diagnostic_criteria_summary", "")
                if test_name:
                    next_steps.append(f"**{test_name}**: {desc}")

            raw_score = math.exp(log_likelihood) * adjusted_prior
            
            # Finalize trace for this disease
            math_trace["log_likelihood"] = log_likelihood
            math_trace["likelihood"] = math.exp(log_likelihood)
            math_trace["raw_score"] = raw_score

            results[d_id] = {
                "name": name, 
                "raw_score": raw_score, 
                "pre_test": adjusted_prior,
                "drivers": drivers,
                "penalties": penalties,
                "missing_hallmarks": missing_hallmarks,
                "next_steps": next_steps,
                "math_trace": math_trace
            }
            total_posterior_weight += raw_score

        differential_list = []
        for d_id, diagnostics in results.items():
            normalized_prob = (diagnostics["raw_score"] / total_posterior_weight) if total_posterior_weight > 0 else 0.0
            
            # Add the total denominator weight to the trace so UI can show the normalization
            diagnostics["math_trace"]["total_weight"] = total_posterior_weight
            
            differential_list.append({
                "id": d_id,
                "name": diagnostics["name"],
                "pre_test_probability": f"{diagnostics['pre_test'] * 100:.2f}%",
                "posterior_probability": f"{normalized_prob * 100:.2f}%",
                "drivers": diagnostics["drivers"],
                "penalties": diagnostics["penalties"],
                "missing_hallmarks": diagnostics["missing_hallmarks"],
                "next_steps": diagnostics["next_steps"],
                "math_trace": diagnostics["math_trace"]
            })

        return sorted(differential_list, key=lambda x: float(x["posterior_probability"].replace('%','')), reverse=True)