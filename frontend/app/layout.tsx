
import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "ASFR: RiceLens AI",
  description:
    "Research prototype for image-level rice disease classification using Asymmetric Spatial Feature Routing, Grad-CAM, and spatial-gate visualization.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
