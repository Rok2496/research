from datetime import datetime
import json

class ReportService:
    def __init__(self):
        self.report_template = {
            'header': {
                'hospital_name': 'CheXpert Medical Center',
                'date': None,
                'patient_id': None
            },
            'analysis': {
                'findings': [],
                'confidence_scores': {},
                'recommendations': []
            }
        }

    def generate_report(self, patient_id, analysis_results):
        try:
            report = self.report_template.copy()
            report['header']['date'] = datetime.now().isoformat()
            report['header']['patient_id'] = patient_id
            
            report['analysis']['findings'] = self._process_findings(analysis_results)
            report['analysis']['confidence_scores'] = analysis_results
            report['analysis']['recommendations'] = self._generate_recommendations(analysis_results)
            
            return report
        except Exception as e:
            raise Exception(f"Report generation error: {str(e)}")

    def _process_findings(self, results):
        findings = []
        for condition, probability in results.items():
            if probability > 0.5:
                findings.append({
                    'condition': condition,
                    'probability': probability,
                    'severity': self._determine_severity(probability)
                })
        return findings

    def _determine_severity(self, probability):
        if probability > 0.8:
            return 'High'
        elif probability > 0.5:
            return 'Medium'
        return 'Low'

    def _generate_recommendations(self, results):
        recommendations = []
        for condition, probability in results.items():
            if probability > 0.8:
                recommendations.append(f"Immediate follow-up recommended for {condition}")
            elif probability > 0.5:
                recommendations.append(f"Regular monitoring recommended for {condition}")
        return recommendations
