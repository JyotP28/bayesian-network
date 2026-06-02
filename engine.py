
import json
import math

class VetBayesianEngine:
    def __init__(self, json_path):
        with open(json_path, 'r') as f:
            self.data = json.load(f)
        
        self.tier_probs = {
            "common_tier_60_to_100_percent_occurrence": 0.80,
            "less_common_tier_20_to_59_percent_occurrence": 0.40,
            "uncommon_or_rare_tier_1_to_19_percent_occurrence": 0.15
        }
        self.background_noise = 0.10

        # demo of the ontology mapping
        self.concept_groups = [
            {"weight_loss", "underweight", "poor_muscle_mass", "cachexia"},
            {"abnormal_appetite", "polyphagia", "inappetence", "anorexia", "decreased_appetite"},
            {"lethargy", "weakness", "fatigue", "decreased_energy", "depression"},
            {"vomiting", "regurgitation", "emesis"},
            {"increased_alkaline_phosphatase", "elevated_alp", "increased_alp"},
            {"urine_specific_gravity_less_than_1_020", "decreased_urine_specific_gravity", "isosthenuria"}
        ]
        
        # fast look-up dictionary
        self.synonym_map = {}
        for group in self.concept_groups:
            group_tuple = tuple(sorted(group)) 
            for term in group:
                self.synonym_map[term] = group_tuple

    def calculate_differentials(self, patient_signalment, active_findings):
        results = {}
        total_posterior_weight = 0.0

        for disease in self.data["diseases"]:
            if "disease_metadata" not in disease: continue
            meta = disease["disease_metadata"]
            d_id = meta["disease_id"]
            name = meta["official_name"]
            
            p_disease = meta.get("estimated_population_base_prevalence")
            sig_mults = disease.get("signalment_risk_multipliers", {})
            
            age_mult = sig_mults.get("age_brackets", {}).get(patient_signalment.get('age'), 1.0)
            sex_mult = sig_mults.get("biological_sex_and_status", {}).get(patient_signalment.get('sex'), 1.0)
            breed_mult = sig_mults.get("predisposed_breeds", {}).get(patient_signalment.get('breed'), 
                            sig_mults.get("predisposed_breeds", {}).get("default_all_other_breeds", 1.0))
            
            # handle missing textbook data and information
            if not isinstance(p_disease, (int, float)):
                adjusted_prior = 1.0
                trace_base_prevalence = "Missing (Forced 1.0)"
            else:
                adjusted_prior = p_disease * age_mult * sex_mult * breed_mult
                trace_base_prevalence = p_disease

            math_trace = {
                "base_prevalence": trace_base_prevalence,
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
            processed_concept_groups = set()

            for finding in active_findings:
                eq_group = self.synonym_map.get(finding, (finding,))
                
                if eq_group in processed_concept_groups:
                    continue 
                
                processed_concept_groups.add(eq_group)
                
                match_found = False
                matched_d_term = None
                
                for eq_term in eq_group:
                    if eq_term in all_mapped_findings:
                        match_found = True
                        matched_d_term = eq_term
                        break
                
                if match_found:
                    tier = all_mapped_findings[matched_d_term]
                    p_finding_given_d = self.tier_probs.get(tier, self.background_noise)
                    tier_display = "Classic" if "common_tier" in tier else "Expected" if "less_common" in tier else "Possible"
                    
                    display_name = matched_d_term.replace('_', ' ').title()
                    if matched_d_term != finding:
                        display_name += f" (via {finding.replace('_', ' ').title()})"
                        
                    drivers.append(f"• **{display_name}** ({tier_display})")
                    math_trace["findings"].append({"name": display_name, "tier": tier_display, "prob": p_finding_given_d, "log_prob": math.log(p_finding_given_d)})
                else:
                    p_finding_given_d = self.background_noise
                    display_name = finding.replace('_', ' ').title()
                    penalties.append(f"• **{display_name}**")
                    math_trace["findings"].append({"name": display_name, "tier": "Unmatched Penalty", "prob": p_finding_given_d, "log_prob": math.log(p_finding_given_d)})
                
                log_likelihood += math.log(p_finding_given_d)

            missing_hallmarks = []
            for d_finding, tier in all_mapped_findings.items():
                if "common_tier" in tier:
                    eq_group = self.synonym_map.get(d_finding, (d_finding,))
                    if eq_group not in processed_concept_groups:
                        missing_hallmarks.append(f"• **{d_finding.replace('_', ' ').title()}**")
                        processed_concept_groups.add(eq_group) 

            next_steps = []
            for test in disease.get("specific_confirmatory_diagnostic_tests", []):
                test_name = test.get("test_name", "").replace('_', ' ').title()
                desc = test.get("textbook_diagnostic_criteria_summary", "")
                if test_name:
                    next_steps.append(f"**{test_name}**: {desc}")

            raw_score = math.exp(log_likelihood) * adjusted_prior
            
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
            diagnostics["math_trace"]["total_weight"] = total_posterior_weight
            
            # safeguard for missing pre-test probabilities for the UI display
            pre_test_display = f"{diagnostics['pre_test'] * 100:.2f}%" if isinstance(diagnostics['pre_test'], (int, float)) else "Unknown"

            differential_list.append({
                "id": d_id,
                "name": diagnostics["name"],
                "pre_test_probability": pre_test_display,
                "posterior_probability": f"{normalized_prob * 100:.2f}%",
                "drivers": diagnostics["drivers"],
                "penalties": diagnostics["penalties"],
                "missing_hallmarks": diagnostics["missing_hallmarks"],
                "next_steps": diagnostics["next_steps"],
                "math_trace": diagnostics["math_trace"]
            })

        return sorted(differential_list, key=lambda x: float(x["posterior_probability"].replace('%','')), reverse=True)

    def get_trajectory_data(self, patient_signalment, active_findings):
        """Calculates the posterior probability step-by-step for visualization."""
        state = []
        
        # initialize Baseline
        for disease in self.data["diseases"]:
            if "disease_metadata" not in disease: continue
            meta = disease["disease_metadata"]
            
            p_disease = meta.get("estimated_population_base_prevalence")
            sig_mults = disease.get("signalment_risk_multipliers", {})
            age_mult = sig_mults.get("age_brackets", {}).get(patient_signalment.get('age'), 1.0)
            sex_mult = sig_mults.get("biological_sex_and_status", {}).get(patient_signalment.get('sex'), 1.0)
            breed_mult = sig_mults.get("predisposed_breeds", {}).get(patient_signalment.get('breed'), 
                            sig_mults.get("predisposed_breeds", {}).get("default_all_other_breeds", 1.0))
            
            if not isinstance(p_disease, (int, float)):
                adjusted_prior = 1.0
            else:
                adjusted_prior = p_disease * age_mult * sex_mult * breed_mult

            symptoms = disease.get("clinical_history_and_physical_exam_symptoms", {})
            labs = disease.get("routine_laboratory_abnormalities_cbc_chem_ua", {})
            all_mapped = {}
            for tier, items in symptoms.items():
                if isinstance(items, list):
                    for item in items: all_mapped[item] = tier
            for tier, items in labs.items():
                if isinstance(items, list):
                    for item in items: all_mapped[item] = tier

            state.append({
                "name": meta["official_name"],
                "prior": adjusted_prior,
                "log_likelihood": 0.0,
                "mapped_findings": all_mapped,
                "processed_groups": set()
            })

        history = []
        
        def _record_step(step_name):
            total_weight = 0.0
            for d in state:
                d["raw_score"] = d["prior"] * math.exp(d["log_likelihood"])
                total_weight += d["raw_score"]
            
            step_record = {"Step": step_name}
            for d in state:
                prob = (d["raw_score"] / total_weight * 100) if total_weight > 0 else 0.0
                step_record[d["name"]] = prob
            history.append(step_record)

        _record_step("Baseline (Signalment)")

        # process teh findings sequentially
        for finding in active_findings:
            eq_group = self.synonym_map.get(finding, (finding,))
            
            for d in state:
                if eq_group in d["processed_groups"]:
                    continue 
                
                d["processed_groups"].add(eq_group)
                
                match_found = False
                matched_term = None
                for eq_term in eq_group:
                    if eq_term in d["mapped_findings"]:
                        match_found = True
                        matched_term = eq_term
                        break
                
                if match_found:
                    tier = d["mapped_findings"][matched_term]
                    prob = self.tier_probs.get(tier, self.background_noise)
                else:
                    prob = self.background_noise
                    
                d["log_likelihood"] += math.log(prob)
            
            _record_step(finding.replace('_', ' ').title())
        
        return history