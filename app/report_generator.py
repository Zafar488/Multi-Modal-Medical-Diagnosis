from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
from datetime import datetime
import yaml
import logging

logger = logging.getLogger(__name__)

# Project root directory (one level up from app/)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class MedicalReportGenerator:
    """Generate professional PDF medical reports"""

    def __init__(self, config_path=None):
        if config_path is None:
            config_path = os.path.join(PROJECT_ROOT, 'config', 'config.yaml')
        self.config = self.load_config(config_path)
        self.styles = getSampleStyleSheet()
        self.custom_styles = self.create_custom_styles()
        output_dir = self.config.get('report', {}).get('output_dir', 'reports/generated')
        if not os.path.isabs(output_dir):
            output_dir = os.path.join(PROJECT_ROOT, output_dir)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def load_config(self, config_path):
        """Load configuration"""
        try:
            with open(config_path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Could not load config: {e}")
            return {}

    def create_custom_styles(self):
        """Create custom styles for the report"""
        styles = {
            'title': ParagraphStyle(
                'CustomTitle',
                parent=self.styles['Heading1'],
                fontSize=24,
                textColor=colors.HexColor('#1a5f7a'),
                alignment=TA_CENTER,
                spaceAfter=30
            ),
            'heading': ParagraphStyle(
                'CustomHeading',
                parent=self.styles['Heading2'],
                fontSize=16,
                textColor=colors.HexColor('#2E86AB'),
                spaceAfter=12
            ),
            'subheading': ParagraphStyle(
                'CustomSubHeading',
                parent=self.styles['Heading3'],
                fontSize=14,
                textColor=colors.HexColor('#3A7CA5'),
                spaceAfter=8
            ),
            'body': ParagraphStyle(
                'CustomBody',
                parent=self.styles['Normal'],
                fontSize=11,
                spaceAfter=6
            ),
            'risk_high': ParagraphStyle(
                'RiskHigh',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#DC3545'),
                alignment=TA_CENTER,
                spaceAfter=12
            ),
            'risk_medium': ParagraphStyle(
                'RiskMedium',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#FFC107'),
                alignment=TA_CENTER,
                spaceAfter=12
            ),
            'risk_low': ParagraphStyle(
                'RiskLow',
                parent=self.styles['Normal'],
                fontSize=14,
                textColor=colors.HexColor('#28A745'),
                alignment=TA_CENTER,
                spaceAfter=12
            ),
            'footer': ParagraphStyle(
                'Footer',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#6C757D'),
                alignment=TA_CENTER
            ),
            'disclaimer': ParagraphStyle(
                'Disclaimer',
                parent=self.styles['Normal'],
                fontSize=9,
                textColor=colors.HexColor('#DC3545'),
                alignment=TA_CENTER
            )
        }
        return styles

    def generate_report(self, patient_name: str, results: dict,
                        symptoms: dict = None, filename: str = None) -> str:
        """Generate comprehensive medical report"""

        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f'medical_report_{timestamp}.pdf'

        filepath = os.path.join(self.output_dir, filename)

        doc = SimpleDocTemplate(
            filepath,
            pagesize=A4,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=72
        )

        story = []

        # Header
        story.append(self._create_header())
        story.append(Spacer(1, 0.2 * inch))

        # Title
        story.append(Paragraph("Medical Diagnosis Report", self.custom_styles['title']))
        story.append(Spacer(1, 0.2 * inch))

        # Patient Information
        story.append(self._create_patient_info(patient_name))
        story.append(Spacer(1, 0.3 * inch))

        # Risk Assessment — returns list, use extend
        story.extend(self._create_risk_assessment(results))
        story.append(Spacer(1, 0.2 * inch))

        # Results — returns list, use extend
        story.extend(self._create_results_section(results))
        story.append(Spacer(1, 0.2 * inch))

        # Symptoms
        if symptoms:
            story.extend(self._create_symptoms_section(symptoms))
            story.append(Spacer(1, 0.2 * inch))

        # Recommendations — returns list, use extend
        story.extend(self._create_recommendations(results))
        story.append(Spacer(1, 0.3 * inch))

        # Health Tips — returns list, use extend
        story.extend(self._create_health_tips())
        story.append(Spacer(1, 0.3 * inch))

        # Footer with Disclaimer — returns list, use extend
        story.extend(self._create_footer())

        # Build PDF
        doc.build(story)
        logger.info(f"✅ Report generated: {filepath}")
        return filepath

    def _create_header(self):
        """Create report header (returns single Paragraph)"""
        company = self.config.get('report', {}).get('company_name', 'AI Medical Diagnosis System')
        header = Paragraph(
            f"<b>{company}</b><br/>"
            f"<font size='10'>AI-Powered Multi-Modal Healthcare Decision Support</font>",
            self.custom_styles['body']
        )
        return header

    def _create_patient_info(self, patient_name):
        """Create patient information section (returns single Table)"""
        data = [
            ['Report ID:', f'MED-{datetime.now().strftime("%Y%m%d")}-{datetime.now().strftime("%H%M%S")}'],
            ['Patient Name:', patient_name],
            ['Date:', datetime.now().strftime('%B %d, %Y at %I:%M %p')],
            ['Generated By:', 'AI Medical Diagnosis System'],
            ['Version:', self.config.get('app', {}).get('version', '1.0.0')]
        ]

        table = Table(data, colWidths=[2.5 * inch, 3.5 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#1a5f7a')),
            ('TEXTCOLOR', (1, 0), (1, -1), colors.black),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#E8F4F8')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))

        return table

    def _create_risk_assessment(self, results):
        """Create risk assessment section (returns list of flowables)"""
        elements = []
        elements.append(Paragraph("Risk Assessment", self.custom_styles['heading']))

        overall_risk = results.get('overall_risk', 'Low')

        risk_style = self.custom_styles.get(f'risk_{overall_risk.lower()}',
                                            self.custom_styles['body'])
        risk_text = f"Overall Risk Level: {overall_risk.upper()}"
        elements.append(Paragraph(risk_text, risk_style))

        risk_factors = []
        if results.get('xray') and results['xray'].get('status') == 'Abnormal':
            risk_factors.append('Abnormal X-Ray findings')
        if results.get('skin') and results['skin'].get('status') == 'Malignant':
            risk_factors.append('Malignant skin lesion detected')
        if results.get('symptom'):
            high_risk = ['Pneumonia', 'COVID-19']
            if results['symptom'].get('disease') in high_risk:
                risk_factors.append(f'{results["symptom"]["disease"]} diagnosis')

        if risk_factors:
            elements.append(Paragraph("Risk Factors:", self.custom_styles['subheading']))
            for factor in risk_factors:
                elements.append(Paragraph(f"- {factor}", self.custom_styles['body']))
        else:
            elements.append(Paragraph("No significant risk factors identified.",
                                      self.custom_styles['body']))

        return elements

    def _create_results_section(self, results):
        """Create results section with all modalities (returns list of flowables)"""
        elements = []
        elements.append(Paragraph("Diagnosis Results", self.custom_styles['heading']))

        # X-Ray Results
        if results.get('xray'):
            elements.append(Paragraph("X-Ray Analysis", self.custom_styles['subheading']))
            xray = results['xray']
            xray_data = [
                ['Finding:', xray.get('result', 'N/A')],
                ['Confidence:', f"{xray.get('confidence', 0):.1%}"],
                ['Status:', xray.get('status', 'N/A')]
            ]
            elements.append(self._create_result_table(xray_data))
            elements.append(Spacer(1, 0.1 * inch))

        # Skin Results
        if results.get('skin'):
            elements.append(Paragraph("Skin Cancer Analysis", self.custom_styles['subheading']))
            skin = results['skin']
            skin_data = [
                ['Finding:', skin.get('result', 'N/A')],
                ['Confidence:', f"{skin.get('confidence', 0):.1%}"],
                ['Status:', skin.get('status', 'N/A')]
            ]
            elements.append(self._create_result_table(skin_data))
            elements.append(Spacer(1, 0.1 * inch))

        # Symptom Results
        if results.get('symptom'):
            elements.append(Paragraph("Symptom Analysis", self.custom_styles['subheading']))
            symptom = results['symptom']
            symptom_data = [
                ['Predicted Disease:', symptom.get('disease', 'N/A')],
                ['Confidence:', f"{symptom.get('confidence', 0):.1%}"]
            ]
            elements.append(self._create_result_table(symptom_data))

            # Disease probabilities
            if 'all_probabilities' in symptom:
                elements.append(Paragraph("Disease Probabilities:",
                                          self.custom_styles['subheading']))
                prob_data = [['Disease', 'Probability']]
                sorted_probs = sorted(symptom['all_probabilities'].items(),
                                      key=lambda x: x[1], reverse=True)
                for disease, prob in sorted_probs[:5]:
                    prob_data.append([disease, f"{prob:.1%}"])

                prob_table = Table(prob_data, colWidths=[2.5 * inch, 2.5 * inch])
                prob_table.setStyle(TableStyle([
                    ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
                    ('PADDING', (0, 0), (-1, -1), 6),
                ]))
                elements.append(prob_table)

        return elements

    def _create_result_table(self, data):
        """Create a styled result table (returns single Table)"""
        table = Table(data, colWidths=[2 * inch, 4 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (0, -1), colors.HexColor('#2E86AB')),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))
        return table

    def _create_symptoms_section(self, symptoms):
        """Create symptoms section (returns list of flowables)"""
        elements = []
        elements.append(Paragraph("Symptoms Reported", self.custom_styles['heading']))

        data = [['Symptom', 'Present']]
        for symptom, present in symptoms.items():
            status = 'Yes' if present else 'No'
            data.append([symptom.replace('_', ' '), status])

        table = Table(data, colWidths=[2.5 * inch, 2.5 * inch])
        table.setStyle(TableStyle([
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#2E86AB')),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#DEE2E6')),
            ('PADDING', (0, 0), (-1, -1), 6),
        ]))

        elements.append(table)
        return elements

    def _create_recommendations(self, results):
        """Create recommendations section (returns list of flowables)"""
        elements = []
        elements.append(Paragraph("Recommendations", self.custom_styles['heading']))

        recommendations = results.get('recommendations',
                                      ['Consult healthcare professional for proper diagnosis'])

        if recommendations:
            for i, rec in enumerate(recommendations, 1):
                elements.append(Paragraph(f"{i}. {rec}", self.custom_styles['body']))
        else:
            elements.append(Paragraph("No specific recommendations.", self.custom_styles['body']))

        return elements

    def _create_health_tips(self):
        """Create general health tips section (returns list of flowables)"""
        elements = []
        elements.append(Paragraph("General Health Tips", self.custom_styles['heading']))

        tips = [
            'Maintain a balanced diet rich in fruits, vegetables, and whole grains',
            'Engage in regular physical activity (30 minutes, 5 days a week)',
            'Stay hydrated by drinking 8-10 glasses of water daily',
            'Ensure 7-8 hours of quality sleep each night',
            'Practice stress management through meditation or yoga',
            'Avoid smoking and limit alcohol consumption',
            'Schedule regular health check-ups',
            'Take medications as prescribed by healthcare provider'
        ]

        for i, tip in enumerate(tips, 1):
            elements.append(Paragraph(f"{i}. {tip}", self.custom_styles['body']))

        return elements

    def _create_footer(self):
        """Create footer with disclaimer (returns list of flowables)"""
        elements = []

        disclaimer_text = (
            "<b>IMPORTANT DISCLAIMER:</b><br/>"
            "This report is generated by an AI-assisted diagnostic system and is intended "
            "for informational and educational purposes only. It does not constitute medical "
            "advice and should not replace professional medical consultation. "
            "Always consult with qualified healthcare providers for proper diagnosis and treatment."
        )
        elements.append(Paragraph(disclaimer_text, self.custom_styles['disclaimer']))
        elements.append(Spacer(1, 0.1 * inch))

        footer_text = (
            f"Generated on: {datetime.now().strftime('%B %d, %Y at %I:%M %p')} | "
            f"Report ID: MED-{datetime.now().strftime('%Y%m%d')}-{datetime.now().strftime('%H%M%S')}"
        )
        elements.append(Paragraph(footer_text, self.custom_styles['footer']))

        return elements