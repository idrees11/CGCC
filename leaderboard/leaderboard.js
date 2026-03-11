// leaderboard.js
document.addEventListener('DOMContentLoaded', function() {
    fetch('leaderboard.csv')
        .then(response => response.text())
        .then(csvText => {
            const rows = csvText.trim().split('\n');
            const headers = rows[0].split(',');
            const data = rows.slice(1).map(row => row.split(','));

            // Update last updated time (using the current date)
            const now = new Date();
            document.getElementById('last-updated').textContent =
                `Last updated: ${now.toLocaleString('en-US', { month: 'long', day: 'numeric', year: 'numeric' })} at ${now.toLocaleTimeString()}`;

            const tbody = document.getElementById('table-body');
            tbody.innerHTML = '';

            data.forEach((row, index) => {
                const rank = row[0];          // rank
                const team = row[1];           // team_name
                const f1Ideal = parseFloat(row[2]).toFixed(4);
                const f1Pert = parseFloat(row[3]).toFixed(4);
                const gap = parseFloat(row[4]).toFixed(4);

                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td class="rank">${rank}</td>
                    <td class="team-name">${team}</td>
                    <td class="score primary-score">${f1Ideal}</td>
                    <td class="score primary-score">${f1Pert}</td>
                    <td class="score">${gap}</td>
                `;
                tbody.appendChild(tr);
            });
        })
        .catch(error => {
            console.error('Error loading CSV:', error);
            document.getElementById('table-body').innerHTML = `
                <tr><td colspan="5" class="empty">Failed to load leaderboard data.</td></tr>
            `;
        });
});
