import { useEffect, useState } from "react";

export const Usage = () => {
  const [usage, setUsage] = useState(0);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchUsage = async () => {
      // get search params from url
      const searchParams = new URLSearchParams(window.location.search);
      const machineName = searchParams.get("machineName");
      const modalSession = searchParams.get("modalSession");
      const url = `https://ruucm-comfy-assets.vercel.app/machine-usage?machineName=${machineName}&modalSession=${modalSession}`;

      try {
        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Error: ${response.status}`);
        }
        const data = await response.json();
        const usageInUsd = data.usageInUsd || 0;
        setUsage(usageInUsd);
      } catch (error: any) {
        console.log("error", error);
        console.log("error.message", error.message);
        setError(error.message);
      }
    };

    fetchUsage();
  }, []);

  return <div>{error ? <></> : <>Usage: ${usage.toFixed(2)} USD</>}</div>;
};
