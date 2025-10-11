document.addEventListener('DOMContentLoaded', () => {
    const journalPanel = document.getElementById('journal-panel-content');

    function formatTimestamp(isoString) {
        if (!isoString) return 'N/A';
        const date = new Date(isoString);
        return date.toLocaleString('en-US', { 
            year: '2-digit', month: '2-digit', day: '2-digit', 
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false 
        });
    }

    async function updateJournal() {
        if (!journalPanel) return;

        try {
            const response = await fetch('/api/journal/trades?limit=50');
            if (!response.ok) {
                journalPanel.innerHTML = '<tr><td colspan="8" class="muted">Error fetching journal data.</td></tr>';
                return;
            }
            const trades = await response.json();

            if (trades.length === 0) {
                journalPanel.innerHTML = '<tr><td colspan="8" class="muted">No trades recorded yet.</td></tr>';
                return;
            }

            const rows = trades.map(trade => `
                <tr>
                    <td class="muted">${formatTimestamp(trade.timestamp)}</td>
                    <td>${trade.symbol}</td>
                    <td class="${trade.direction.toLowerCase() === 'buy' ? 'good' : 'bad'}">${trade.direction.toUpperCase()}</td>
                    <td>${trade.volume.toFixed(2)}</td>
                    <td>${trade.entry_price ? trade.entry_price.toFixed(5) : 'N/A'}</td>
                    <td>${trade.stop_loss ? trade.stop_loss.toFixed(5) : 'N/A'}</td>
                    <td>${trade.take_profit ? trade.take_profit.toFixed(5) : 'N/A'}</td>
                    <td class="muted rationale">${trade.rationale || ''}</td>
                </tr>
            `).join('');

            journalPanel.innerHTML = rows;

        } catch (error) {
            console.error('Failed to update journal:', error);
            journalPanel.innerHTML = '<tr><td colspan="8" class="muted">Failed to load journal.</td></tr>';
        }
    }

    // Initial update and then poll every 15 seconds
    updateJournal();
    setInterval(updateJournal, 15000);
});
