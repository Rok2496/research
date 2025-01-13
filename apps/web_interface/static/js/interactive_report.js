class InteractiveReport {
    constructor() {
        this.reportData = null;
        this.init();
    }

    init() {
        this.setupEventListeners();
    }

    setupEventListeners() {
        const generateBtn = document.getElementById('generateReport');
        if (generateBtn) {
            generateBtn.addEventListener('click', () => this.generateReport());
        }
    }

    async generateReport() {
        try {
            const response = await fetch('/api/report/generate', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(this.reportData)
            });
            
            const result = await response.json();
            this.displayReport(result);
        } catch (error) {
            console.error('Error generating report:', error);
        }
    }

    displayReport(data) {
        const container = document.getElementById('reportContainer');
        if (container) {
            container.innerHTML = this.formatReport(data);
        }
    }

    formatReport(data) {
        return `
            <div class="report-section">
                <h3>Analysis Results</h3>
                <div class="results-container">
                    ${this.formatResults(data.results)}
                </div>
            </div>
        `;
    }

    formatResults(results) {
        return results.map(result => `
            <div class="result-item">
                <h4>${result.title}</h4>
                <p>${result.description}</p>
                <div class="confidence-score">
                    Confidence: ${result.confidence}%
                </div>
            </div>
        `).join('');
    }
}

const interactiveReport = new InteractiveReport();
