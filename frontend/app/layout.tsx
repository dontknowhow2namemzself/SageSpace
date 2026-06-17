import type { Metadata } from 'next'
import { EB_Garamond } from 'next/font/google'
import './globals.css'

const garamond = EB_Garamond({
  subsets: ['latin'],
  variable: '--font-serif',
  weight: ['400', '500', '600'],
})

export const metadata: Metadata = {
  title: "Scholar's Den",
  description: 'Talk with the wisdom inside every book.',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={`${garamond.variable} bg-stone-950 text-stone-100 min-h-screen antialiased`}>
        {children}
      </body>
    </html>
  )
}
