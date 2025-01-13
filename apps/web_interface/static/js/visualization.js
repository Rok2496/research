class Visualization {
    constructor() {
        this.charts = {};
        this.init();
    }

    init() {
        this.initializeCharts();
        this.setupEventListeners();
    }

    initializeCharts() {
        const ctx = document.getElementById('analysisChart');
        if (ctx) {
            this.charts.analysis = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Analysis Accuracy',
                        data: [],
                        borderColor: 'rgb(75, 192, 192)',
                        tension: 0.1
                    }]
                },
                options: {
                    responsive: true,
                    plugins: {
                        title: {
                            display: true,
                            text: 'Analysis Performance Over Time'
                        }
                    }
                }
            });
        }
    }

    setupEventListeners() {
        document.addEventListener('DOMContentLoaded', () => {
            this.loadInitialData();
        });
    }

    async loadInitialData() {
        try {
            const response = await fetch('/api/analysis/stats');
            const data = await response.json();
            this.updateCharts(data);
        } catch (error) {
            console.error('Error loading data:', error);
        }
    }

    updateCharts(data) {
        if (this.charts.analysis) {
            this.charts.analysis.data.labels = data.labels;
            this.charts.analysis.data.datasets[0].data = data.values;
            this.charts.analysis.update();
        }
    }
}

const visualization = new Visualization();
