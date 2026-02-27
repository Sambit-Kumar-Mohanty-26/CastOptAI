"use client";

import { useState, useEffect, useCallback, useRef } from "react";
import axios from "axios";
import type { OptResult, WeatherData } from "@/types";

import Sidebar from "@/components/Sidebar";
import Header from "@/components/Header";
import InputPanel from "@/components/InputPanel";
import EmptyState from "@/components/EmptyState";
import ResultsPanel from "@/components/ResultsPanel";
import ParetoOptimizerPanel from "@/components/phase2/ParetoOptimizer";
import RiskAssessmentPanel from "@/components/phase2/RiskAssessment";

const AI_SERVICE_URL = process.env.NEXT_PUBLIC_AI_SERVICE_URL || "http://localhost:8000";
const WEATHER_API_KEY = process.env.NEXT_PUBLIC_OPENWEATHER_API_KEY || "";

const DEMO_WEATHER: Record<string, WeatherData> = {
  Delhi: { temp: 10, humidity: 45, desc: "haze" },
  Mumbai: { temp: 32, humidity: 80, desc: "partly cloudy" },
  Chennai: { temp: 30, humidity: 75, desc: "clear sky" },
  Kolkata: { temp: 22, humidity: 65, desc: "mist" },
  Bangalore: { temp: 26, humidity: 55, desc: "scattered clouds" },
  Hyderabad: { temp: 28, humidity: 50, desc: "clear sky" },
  Pune: { temp: 25, humidity: 48, desc: "few clouds" },
  Ahmedabad: { temp: 35, humidity: 30, desc: "clear sky" },
};


interface OptContext {
  city: string;
  temp: number;
  humidity: number;
  targetStrength: number;
  targetTime: number;
  timestamp: string;
}

export default function Dashboard() {
  const [city, setCity] = useState("Delhi");
  const [targetTime, setTargetTime] = useState(24);
  const [targetStrength, setTargetStrength] = useState(15);
  const [weather, setWeather] = useState<WeatherData>({ temp: null, humidity: null, desc: "Loading..." });
  const [weatherLoading, setWeatherLoading] = useState(false);

  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<OptResult | null>(null);
  const [selectedStrategy, setSelectedStrategy] = useState(0);
  const [error, setError] = useState("");
  const [activeTab, setActiveTab] = useState<"results" | "whatif" | "learn">("results");
  
  // Phase 2 State Variables
  const [paretoData, setParetoData] = useState<any>(null);
  const [riskData, setRiskData] = useState<any>(null);
  const [showPhase2, setShowPhase2] = useState(false);
  const [optContext, setOptContext] = useState<OptContext | null>(null);
  const [paramsChanged, setParamsChanged] = useState(false);

  // City to site ID mapping
  const cityToSiteId: Record<string, string> = {
    "Delhi": "delhi_yard",
    "Mumbai": "mumbai_yard",
    "Chennai": "chennai_yard",
    "Kolkata": "kolkata_yard",
    "Bangalore": "bangalore_yard",
    "Hyderabad": "hyderabad_yard",
    "Pune": "pune_yard",
    "Ahmedabad": "ahmedabad_yard",
  };

  const hasRunRef = useRef(false);
  const autoRunTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const fetchWeather = useCallback(async (selectedCity: string) => {
    setCity(selectedCity);
    if (!WEATHER_API_KEY) {
      setWeather(DEMO_WEATHER[selectedCity] || { temp: 25, humidity: 60, desc: "moderate" });
      return;
    }
    setWeatherLoading(true);
    try {
      const res = await axios.get(
        `https://api.openweathermap.org/data/2.5/weather?q=${encodeURIComponent(selectedCity)}&units=metric&appid=${WEATHER_API_KEY}`
      );
      setWeather({
        temp: Math.round(res.data.main.temp),
        humidity: res.data.main.humidity,
        desc: res.data.weather[0].description,
      });
    } catch {
      setWeather(DEMO_WEATHER[selectedCity] || { temp: 25, humidity: 60, desc: "fetch failed" });
    }
    setWeatherLoading(false);
  }, []);

  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    fetchWeather(city);
  }, []);

  const handleOptimize = useCallback(async () => {
    if (weather.temp === null) {
      setError("Please select a city first.");
      return;
    }
    setLoading(true);
    setError("");
    setParamsChanged(false);
    try {
      const siteId = cityToSiteId[city] || city.toLowerCase().replace(/\s+/g, '_') + '_yard';
      
      const res = await axios.post(`${AI_SERVICE_URL}/optimize`, {
        target_strength: targetStrength,
        target_time: targetTime,
        temp: weather.temp,
        humidity: weather.humidity,
        site_id: siteId,
        use_real_time_data: false,  // Set to true if you want to use real-time data
      });
      if (res.data.status === "success") {
        setResult(res.data);
        setSelectedStrategy(0);
        setActiveTab("results");
        hasRunRef.current = true;
        setError(""); // Clear any previous errors

        setOptContext({
          city,
          temp: weather.temp,
          humidity: weather.humidity!,
          targetStrength,
          targetTime,
          timestamp: new Date().toLocaleTimeString("en-IN", { hour: "2-digit", minute: "2-digit", second: "2-digit" }),
        });
      } else {
        // Provide more helpful error messages
        let errorMessage = res.data.message || "Optimization failed.";
        if (errorMessage.includes("too aggressive")) {
          errorMessage = `Could not find optimal recipe. Try: lower strength target (${Math.max(5, targetStrength - 5)} MPa) or longer time (${targetTime + 12} hours)`;
        }
        setError(errorMessage);
      }
    } catch {
      setError("Cannot connect to CastOpt AI service. Ensure FastAPI is running on port 8000.");
    }
    setLoading(false);
  }, [weather, targetStrength, targetTime, city]);

  // Phase 2 Functions
  const handleParetoOptimize = useCallback(async (weights: any) => {
    if (weather.temp === null) {
      setError("Please select a city first.");
      return;
    }
    
    setLoading(true);
    try {
      const siteId = cityToSiteId[city] || city.toLowerCase().replace(/\s+/g, '_') + '_yard';
      
      const res = await axios.post(`${AI_SERVICE_URL}/phase2/pareto-optimization`, {
        target_strength: targetStrength,
        target_time: targetTime,
        site_id: siteId,
        objective_weights: weights,
        constraint_relaxation: 0.1
      });
      
      if (res.data.status === "success") {
        setParetoData(res.data.data.visualization_data.solutions);
        setRiskData(null); // Clear risk data for fresh analysis
        setShowPhase2(true);
        setActiveTab("results");
      } else {
        setError(res.data.message || "Pareto optimization failed.");
      }
    } catch (err) {
      setError("Cannot connect to CastOpt AI Phase 2 service.");
      console.error("Pareto optimization error:", err);
    }
    setLoading(false);
  }, [weather, targetStrength, targetTime, city]);

  const handleRiskAssessment = useCallback(async () => {
    if (!result || !result.strategies || result.strategies.length === 0) {
      setError("Please run optimization first.");
      return;
    }
    
    setLoading(true);
    try {
      const selectedRecipe = result.strategies[selectedStrategy].recommended_recipe;
      
      const res = await axios.post(`${AI_SERVICE_URL}/phase2/risk-assessment`, {
        recipe: selectedRecipe,
        environmental_conditions: {
          temperature: weather.temp,
          humidity: weather.humidity
        },
        target_strength: targetStrength
      });
      
      if (res.data.status === "success") {
        setRiskData(res.data.data);
        setShowPhase2(true);
        setActiveTab("results");
      } else {
        setError(res.data.message || "Risk assessment failed.");
      }
    } catch (err) {
      setError("Cannot connect to CastOpt AI risk assessment service.");
      console.error("Risk assessment error:", err);
    }
    setLoading(false);
  }, [result, selectedStrategy, weather, targetStrength]);


  useEffect(() => {

    if (weather.temp === null) return;

    setParamsChanged(true);


    if (autoRunTimerRef.current) clearTimeout(autoRunTimerRef.current);
    autoRunTimerRef.current = setTimeout(() => {
      handleOptimize();
    }, 1500);

    return () => {
      if (autoRunTimerRef.current) clearTimeout(autoRunTimerRef.current);
    };
  }, [city, targetTime, targetStrength, weather.temp, weather.humidity]);

  return (
    <div className="flex h-screen w-full bg-transparent overflow-hidden">
      <Sidebar />
      <div className="flex-1 flex flex-col h-full bg-transparent">
        <Header />
        <main className="flex-1 overflow-auto p-4 md:p-8">
          <div className="max-w-[1400px] mx-auto w-full">
            <div className="flex items-center justify-between mb-8 animate-assembly">
              <div>
                <h1 className="text-3xl font-extrabold text-[#0F172A] tracking-tight">Optimization Hub</h1>
                <p className="text-[13px] text-[#64748B] font-medium mt-1">Configure parameters and execute AI mix optimizations.</p>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-8">

              <div className="lg:col-span-4 max-w-[420px] w-full">
                <InputPanel
                  city={city} targetTime={targetTime} targetStrength={targetStrength}
                  weather={weather} weatherLoading={weatherLoading}
                  loading={loading} error={error}
                  onCityChange={fetchWeather}
                  onTargetTimeChange={setTargetTime}
                  onTargetStrengthChange={setTargetStrength}
                  onOptimize={handleOptimize}
                />
              </div>


              <div className="lg:col-span-8">
                {!result && !loading && <EmptyState />}

                {loading && (
                  <div className="h-full flex flex-col items-center justify-center p-12 text-[#94A3B8] animate-pulse-intense">
                    <div className="w-16 h-16 border-4 border-[#E2E8F0] border-t-[#0F172A] rounded-full animate-spin mb-6" />
                    <p className="text-[14px] font-extrabold text-[#0F172A] tracking-wider uppercase">Running AI Simulations...</p>
                    <p className="text-[11px] font-mono-data mt-2 text-[#64748B]">
                      Synthesizing {targetStrength} MPa in {targetTime}h @ {weather.temp}°C / {weather.humidity}% RH — {city}
                    </p>
                  </div>
                )}

                {result && !loading && (
                  <div className="space-y-6">
                    {/* Standard Results Panel */}
                    <ResultsPanel
                      result={result}
                      selectedStrategy={selectedStrategy}
                      onSelectStrategy={setSelectedStrategy}
                      activeTab={activeTab}
                      onTabChange={setActiveTab}
                      weather={{ temp: weather.temp, humidity: weather.humidity }}
                      targetStrength={targetStrength}
                      targetTime={targetTime}
                      optContext={optContext}
                      paramsChanged={paramsChanged}
                    />
                    
                    {/* Phase 2 Controls */}
                    <div className="flex flex-wrap gap-4 pt-4">
                      <button 
                        onClick={handleRiskAssessment}
                        disabled={loading}
                        className="px-4 py-2 bg-orange-500 hover:bg-orange-600 text-white text-sm font-bold rounded transition-colors disabled:opacity-50"
                      >
                        Analyze Risk Profile
                      </button>
                    </div>
                    
                    {/* Phase 2 Components */}
                    <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                      <ParetoOptimizerPanel 
                        onOptimize={handleParetoOptimize}
                        paretoData={paretoData}
                      />
                      <RiskAssessmentPanel 
                        riskData={riskData}
                      />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}