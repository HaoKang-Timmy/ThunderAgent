export const metadata = {
    title: "ThunderAgent - Fast, Simple, and Robust Agentic Inference",
    description:
      "ThunderAgent is a fast, simple, and robust program-aware agentic inference system achieving 1.5-3.6x throughput improvements.",
  };
  
  export default function RootLayout({
    children,
  }: {
    children: React.ReactNode;
  }) {
    return (
      <html lang="en">
        <body style={{ margin: 0, padding: 0, overflow: "hidden" }}>
          {children}
        </body>
      </html>
    );
  }
  