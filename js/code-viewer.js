// Code Viewer with Syntax Highlighting
class CodeViewer {
	constructor() {
		this.modal = null
		this.init()
	}

	init() {
		// Create modal HTML if it doesn't exist
		if (!document.getElementById("code-modal")) {
			const modalHTML = `
        <div id="code-modal" class="code-modal">
          <div class="code-modal-content">
            <span class="code-modal-close">&times;</span>
            <h2 id="code-modal-title"></h2>
            <pre><code id="code-modal-body" class="language-python"></code></pre>
          </div>
        </div>
      `
			document.body.insertAdjacentHTML("beforeend", modalHTML)
			this.modal = document.getElementById("code-modal")

			// Close button functionality
			document
				.querySelector(".code-modal-close")
				.addEventListener("click", () => this.close())

			// Click outside modal to close
			window.addEventListener("click", (event) => {
				if (event.target === this.modal) {
					this.close()
				}
			})
		}
	}

	async open(fileUrl, fileName) {
		try {
			// Use XMLHttpRequest for better compatibility
			const code = await new Promise((resolve, reject) => {
				const xhr = new XMLHttpRequest()
				xhr.open("GET", fileUrl, true)
				xhr.onload = function () {
					if (xhr.status === 200) {
						resolve(xhr.responseText)
					} else {
						reject(new Error(`HTTP Error: ${xhr.status}`))
					}
				}
				xhr.onerror = function () {
					reject(new Error("Network error"))
				}
				xhr.send()
			})

			document.getElementById("code-modal-title").textContent = fileName
			const codeBlock = document.getElementById("code-modal-body")
			codeBlock.textContent = code

			// Apply syntax highlighting
			this.highlightCode(codeBlock)

			this.modal.style.display = "block"
		} catch (error) {
			console.error("Error loading file:", error)
			document.getElementById("code-modal-title").textContent = fileName

			let errorMessage = "Error loading file: " + error.message

			// Check if running locally
			if (window.location.protocol === "file:") {
				errorMessage +=
					"\n\n⚠️  Local Testing Limitation:\nYou're testing locally (file:// protocol). XMLHttpRequest cannot load files locally due to CORS restrictions.\n\nThis will work perfectly once you deploy to your web server!\n\nFile path: " +
					fileUrl
			} else {
				errorMessage += "\n\nMake sure the file exists at: " + fileUrl
			}

			document.getElementById("code-modal-body").textContent =
				errorMessage
			this.modal.style.display = "block"
		}
	}

	close() {
		if (this.modal) {
			this.modal.style.display = "none"
		}
	}

	highlightCode(codeBlock) {
		// Simple display - just escape HTML and show as-is
		let text = codeBlock.textContent

		// Escape HTML special characters
		text = text
			.replace(/&/g, "&amp;")
			.replace(/</g, "&lt;")
			.replace(/>/g, "&gt;")

		codeBlock.innerHTML = text
	}
}

// Initialize on page load
const codeViewer = new CodeViewer()

// Global function to open code files
function openCodeFile(fileUrl, fileName) {
	codeViewer.open(fileUrl, fileName)
}
