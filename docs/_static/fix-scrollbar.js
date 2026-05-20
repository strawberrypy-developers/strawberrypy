window.addEventListener("load", () => {    
    // Force left sidebar back to the top on load
    const sidebar = document.querySelector(".bd-sidebar") || document.getElementById("pst-primary-sidebar");
    if (sidebar) {
        sidebar.scrollTop = 0;
    }
});