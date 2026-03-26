package com.jio.antennachatbot

import android.annotation.SuppressLint
import android.app.Activity
import android.app.AlertDialog
import android.app.DownloadManager
import android.content.Context
import android.content.Intent
import android.net.ConnectivityManager
import android.net.NetworkCapabilities
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.view.KeyEvent
import android.view.View
import android.webkit.*
import android.widget.*
import android.graphics.Color

class MainActivity : Activity() {

    private lateinit var webView: WebView
    private lateinit var progressBar: ProgressBar
    private lateinit var errorLayout: LinearLayout
    private lateinit var loadingLayout: LinearLayout

    private val CHATBOT_URL =
        "https://chatbot-input-database.s3.ap-south-1.amazonaws.com/ui/index.html"

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        try {
            buildUI()
            loadChatbot()
        } catch (e: Exception) {
            showFatalError(e.message ?: "Unknown error")
        }
    }

    @SuppressLint("SetJavaScriptEnabled")
    private fun buildUI() {
        val root = RelativeLayout(this)
        root.setBackgroundColor(Color.WHITE)

        // ── Loading screen ──────────────────────────────────────────────────
        loadingLayout = LinearLayout(this)
        loadingLayout.orientation = LinearLayout.VERTICAL
        loadingLayout.gravity = android.view.Gravity.CENTER
        loadingLayout.setBackgroundColor(Color.WHITE)
        val llParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT,
            RelativeLayout.LayoutParams.MATCH_PARENT
        )
        loadingLayout.layoutParams = llParams

        val loadingSpinner = ProgressBar(this)
        val spinParams = LinearLayout.LayoutParams(120, 120)
        spinParams.bottomMargin = 32
        loadingSpinner.layoutParams = spinParams

        val loadingText = TextView(this)
        loadingText.text = "Loading Jio Antenna Chatbot..."
        loadingText.textSize = 16f
        loadingText.setTextColor(Color.parseColor("#555555"))
        loadingLayout.addView(loadingSpinner)
        loadingLayout.addView(loadingText)
        root.addView(loadingLayout)

        // ── Error screen ───────────────────────────────────────────────────
        errorLayout = LinearLayout(this)
        errorLayout.orientation = LinearLayout.VERTICAL
        errorLayout.gravity = android.view.Gravity.CENTER
        errorLayout.setBackgroundColor(Color.WHITE)
        errorLayout.setPadding(60, 60, 60, 60)
        errorLayout.visibility = View.GONE
        val elParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT,
            RelativeLayout.LayoutParams.MATCH_PARENT
        )
        errorLayout.layoutParams = elParams

        val errorIcon = TextView(this)
        errorIcon.text = "📡"
        errorIcon.textSize = 48f
        errorIcon.gravity = android.view.Gravity.CENTER

        val errorTitle = TextView(this)
        errorTitle.text = "Connection Error"
        errorTitle.textSize = 20f
        errorTitle.setTextColor(Color.parseColor("#d50000"))
        errorTitle.gravity = android.view.Gravity.CENTER
        val etParams = LinearLayout.LayoutParams(
            LinearLayout.LayoutParams.MATCH_PARENT, LinearLayout.LayoutParams.WRAP_CONTENT
        )
        etParams.topMargin = 16; etParams.bottomMargin = 12
        errorTitle.layoutParams = etParams

        val errorMsg = TextView(this)
        errorMsg.text = "Unable to load the chatbot.\nPlease check your internet connection."
        errorMsg.textSize = 14f
        errorMsg.setTextColor(Color.parseColor("#666666"))
        errorMsg.gravity = android.view.Gravity.CENTER

        val retryBtn = Button(this)
        retryBtn.text = "Retry"
        retryBtn.setBackgroundColor(Color.parseColor("#d50000"))
        retryBtn.setTextColor(Color.WHITE)
        val rbParams = LinearLayout.LayoutParams(400, 120)
        rbParams.topMargin = 40
        retryBtn.layoutParams = rbParams
        retryBtn.setOnClickListener {
            errorLayout.visibility = View.GONE
            loadingLayout.visibility = View.VISIBLE
            loadChatbot()
        }

        errorLayout.addView(errorIcon)
        errorLayout.addView(errorTitle)
        errorLayout.addView(errorMsg)
        errorLayout.addView(retryBtn)
        root.addView(errorLayout)

        // ── Progress bar ───────────────────────────────────────────────────
        progressBar = ProgressBar(this, null, android.R.attr.progressBarStyleHorizontal)
        progressBar.id = android.R.id.progress
        progressBar.max = 100
        val pbParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT, 8
        )
        progressBar.layoutParams = pbParams
        progressBar.visibility = View.GONE
        root.addView(progressBar)

        // ── WebView ────────────────────────────────────────────────────────
        webView = WebView(this)
        val wvParams = RelativeLayout.LayoutParams(
            RelativeLayout.LayoutParams.MATCH_PARENT,
            RelativeLayout.LayoutParams.MATCH_PARENT
        )
        wvParams.addRule(RelativeLayout.BELOW, android.R.id.progress)
        webView.layoutParams = wvParams
        webView.visibility = View.GONE

        webView.settings.apply {
            javaScriptEnabled        = true
            domStorageEnabled        = true
            databaseEnabled          = true
            loadWithOverviewMode     = true
            useWideViewPort          = true
            builtInZoomControls      = false
            displayZoomControls      = false
            setSupportZoom(false)
            @Suppress("DEPRECATION")
            mixedContentMode         = WebSettings.MIXED_CONTENT_ALWAYS_ALLOW
            cacheMode                = WebSettings.LOAD_DEFAULT
            // Allow file access for blob downloads
            allowFileAccess          = true
        }

        // ── Download listener — handles CSV downloads via DownloadManager ──
        webView.setDownloadListener { url, userAgent, contentDisposition, mimeType, _ ->
            try {
                val request = DownloadManager.Request(Uri.parse(url)).apply {
                    setMimeType(mimeType)
                    val filename = URLUtil.guessFileName(url, contentDisposition, mimeType)
                    setTitle(filename)
                    setDescription("Downloading $filename")
                    addRequestHeader("User-Agent", userAgent)
                    setNotificationVisibility(
                        DownloadManager.Request.VISIBILITY_VISIBLE_NOTIFY_COMPLETED
                    )
                    setDestinationInExternalPublicDir(
                        Environment.DIRECTORY_DOWNLOADS, filename
                    )
                }
                val dm = getSystemService(Context.DOWNLOAD_SERVICE) as DownloadManager
                dm.enqueue(request)
                Toast.makeText(this, "Downloading CSV to Downloads folder...",
                    Toast.LENGTH_LONG).show()
            } catch (e: Exception) {
                Toast.makeText(this, "Download failed: ${e.message}",
                    Toast.LENGTH_LONG).show()
            }
        }

        webView.webChromeClient = object : WebChromeClient() {
            override fun onProgressChanged(view: WebView, newProgress: Int) {
                progressBar.progress = newProgress
                if (newProgress >= 100) {
                    progressBar.visibility   = View.GONE
                    loadingLayout.visibility = View.GONE
                    webView.visibility       = View.VISIBLE
                } else {
                    progressBar.visibility = View.VISIBLE
                }
            }
        }

        webView.webViewClient = object : WebViewClient() {
            override fun shouldOverrideUrlLoading(
                view: WebView, request: WebResourceRequest
            ): Boolean {
                val url = request.url.toString()
                // Keep S3 and Lambda URLs inside app
                if (url.contains("amazonaws.com") || url.contains("lambda-url")) {
                    view.loadUrl(url)
                    return true
                }
                // Open other links in external browser
                startActivity(Intent(Intent.ACTION_VIEW, Uri.parse(url)))
                return true
            }

            override fun onPageFinished(view: WebView, url: String) {
                super.onPageFinished(view, url)
                loadingLayout.visibility = View.GONE
                webView.visibility       = View.VISIBLE
            }

            @Suppress("DEPRECATION")
            override fun onReceivedError(
                view: WebView, errorCode: Int,
                description: String, failingUrl: String
            ) {
                super.onReceivedError(view, errorCode, description, failingUrl)
                loadingLayout.visibility = View.GONE
                errorLayout.visibility   = View.VISIBLE
                webView.visibility       = View.GONE
            }

            override fun onReceivedSslError(
                view: WebView, handler: SslErrorHandler, error: android.net.http.SslError
            ) {
                handler.proceed() // AWS certs are valid
            }
        }

        root.addView(webView)
        setContentView(root)
    }

    private fun loadChatbot() {
        if (!isNetworkAvailable()) {
            loadingLayout.visibility = View.GONE
            errorLayout.visibility   = View.VISIBLE
            return
        }
        webView.loadUrl(CHATBOT_URL)
    }

    private fun isNetworkAvailable(): Boolean {
        return try {
            val cm = getSystemService(Context.CONNECTIVITY_SERVICE) as ConnectivityManager
            if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.M) {
                val network = cm.activeNetwork ?: return false
                val caps = cm.getNetworkCapabilities(network) ?: return false
                caps.hasCapability(NetworkCapabilities.NET_CAPABILITY_INTERNET)
            } else {
                @Suppress("DEPRECATION")
                cm.activeNetworkInfo?.isConnected == true
            }
        } catch (e: Exception) { true }
    }

    private fun showFatalError(msg: String) {
        try {
            AlertDialog.Builder(this)
                .setTitle("Startup Error")
                .setMessage("Error: $msg\n\nPlease reinstall the app.")
                .setPositiveButton("OK") { _, _ -> finish() }
                .show()
        } catch (e: Exception) { finish() }
    }

    override fun onKeyDown(keyCode: Int, event: KeyEvent?): Boolean {
        if (keyCode == KeyEvent.KEYCODE_BACK && ::webView.isInitialized && webView.canGoBack()) {
            webView.goBack(); return true
        }
        return super.onKeyDown(keyCode, event)
    }

    override fun onPause()  { super.onPause();  if (::webView.isInitialized) webView.onPause() }
    override fun onResume() { super.onResume(); if (::webView.isInitialized) webView.onResume() }
    override fun onDestroy() {
        if (::webView.isInitialized) { webView.stopLoading(); webView.destroy() }
        super.onDestroy()
    }
}
