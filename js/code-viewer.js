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
		// If running from file://, open a local-file picker fallback
		if (window.location.protocol === "file:") {
			this.openLocalPrompt(fileUrl, fileName)
			return
		}

		try {
			// Use XMLHttpRequest for better compatibility when served over HTTP(S)
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

			// Render plain code (no aggressive highlighting)
			this.highlightCode(codeBlock)

			this.modal.style.display = "block"
			// prevent page behind the modal from scrolling while modal is open
			document.body.style.overflow = "hidden"
		} catch (error) {
			console.error("Error loading file:", error)
			document.getElementById("code-modal-title").textContent = fileName
			document.getElementById("code-modal-body").textContent =
				"Error loading file: " +
				error.message +
				"\n\nMake sure the file exists at: " +
				fileUrl
			this.modal.style.display = "block"
			// prevent page behind the modal from scrolling while modal is open
			document.body.style.overflow = "hidden"
		}
	}

	// Show user a prompt to select the file from disk when testing locally
	openLocalPrompt(fileUrl, fileName) {
		document.getElementById("code-modal-title").textContent = fileName
		const body = document.getElementById("code-modal-body")
		body.innerHTML = ""

		const info = document.createElement("div")
		info.style.color = "#222"
		info.style.marginBottom = "10px"
		info.textContent =
			"You're running the page locally. Please select the file from your disk to view it (choose the matching file in the 'codes' folder)."
		body.appendChild(info)

		// Add local file input
		const input = document.createElement("input")
		input.type = "file"
		input.accept = ".py"
		input.style.display = "block"
		input.style.marginBottom = "10px"
		body.appendChild(input)

		// Add small tip to run a local server
		const tip = document.createElement("div")
		tip.style.color = "#666"
		tip.style.fontSize = "0.9rem"
		tip.style.marginTop = "8px"
		tip.innerHTML =
			"Tip: for convenience run a local server and open <code>http://localhost:8000</code>.\nIn PowerShell: <code>python -m http.server 8000</code>"
		body.appendChild(tip)

		input.addEventListener("change", (e) => {
			const file = e.target.files[0]
			if (!file) return
			const reader = new FileReader()
			reader.onload = () => {
				body.textContent = ""
				const pre = document.createElement("pre")
				const codeEl = document.createElement("code")
				codeEl.textContent = reader.result
				pre.appendChild(codeEl)
				body.appendChild(pre)
			}
			reader.readAsText(file)
		})

		this.modal.style.display = "block"
		document.body.style.overflow = "hidden"
	}

	close() {
		if (this.modal) {
			this.modal.style.display = "none"
			// restore page scrolling
			document.body.style.overflow = ""
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
