document.addEventListener('DOMContentLoaded', function () {
    const selectSelected = document.querySelector('.select-selected');
    const selectItems = document.querySelector('.select-items');
    const images = document.querySelectorAll('.visualization-image');

    // Toggle dropdown visibility
    selectSelected.addEventListener('click', () => {
        selectItems.style.display = selectItems.style.display === 'block' ? 'none' : 'block';
    });

    // Close dropdown if clicked outside
    document.addEventListener('click', (event) => {
        if (!selectSelected.contains(event.target) && !selectItems.contains(event.target)) {
            selectItems.style.display = 'none';
        }
    });

    // Show the selected image
    selectItems.addEventListener('click', (event) => {
        if (event.target.tagName === 'DIV') {
            const index = parseInt(event.target.getAttribute('data-index'));
            images.forEach((img, i) => {
                img.classList.toggle('hidden', i !== index);
            });
            selectSelected.textContent = event.target.textContent;
            selectItems.style.display = 'none';
        }
    });
});